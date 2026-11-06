"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import exp, floor
import random

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")
    
    deadline = time.time() + 180.0  # Safety margin
    
    def get_symmetric_indices(size):
        # Independent pixels due to 4-fold symmetry
        # (i, j) where i <= j (transposition symmetry)
        # i <= size-1-i (vertical flip) -> i < size/2
        # j <= size-1-j (horizontal flip) -> j < size/2
        # So i in [0, size//2 - 1] if size even, [0, size//2] if odd?
        # Actually for 32x32:
        # i < 16, j < 16, and i <= j
        # But also need to satisfy all 4 symmetries.
        # The group is D4. The fundamental domain is the triangle i <= j, i < 16, j < 16?
        # Let's verify:
        # Symmetries:
        # 1. (i, j)
        # 2. (size-1-i, j)
        # 3. (i, size-1-j)
        # 4. (size-1-i, size-1-j)
        # 5. (j, i)
        # 6. (size-1-j, i)
        # 7. (j, size-1-i)
        # 8. (size-1-j, size-1-i)
        
        # The canonical representative is the one with minimum lexicographic order or similar.
        # Let's just generate the set of all pixels and then map them to their canonical form.
        # For 32x32, there are 32*32 = 1024 pixels.
        # The number of fundamental pixels is small.
        # i in [0, 15], j in [0, 31] is not right.
        # The region is i <= j and i + j < size? No.
        # The irreducible wedge is 0 <= i <= size/2, 0 <= j <= size/2, i <= j.
        # For size=32: i in [0, 15], j in [0, 15], i <= j.
        # Total = 16*17/2 = 136 pixels.
        
        pixels = []
        for i in range(size // 2):
            for j in range(i, size // 2):
                pixels.append((i, j))
        
        # Map each pixel to the list of pixels it covers
        coverage = {}
        for i, j in pixels:
            group = set()
            group.add((i, j))
            group.add((size-1-i, j))
            group.add((i, size-1-j))
            group.add((size-1-i, size-1-j))
            group.add((j, i))
            group.add((size-1-j, i))
            group.add((j, size-1-i))
            group.add((size-1-j, size-1-i))
            coverage[(i, j)] = group
            
        # Sort for reproducibility
        return pixels, coverage

    pixels, coverage = get_symmetric_indices(size)
    num_pixels = len(pixels)
    
    # Initial state: Dielectric with central air hole
    # Let's try a square hole first, or maybe just the seed rod.
    # The seed is a rod of radius 0.2.
    # Let's start with a design that is known to be good:
    # "Fish-eye" or similar?
    # Let's try a central air square.
    
    # Initialize as all dielectric (1)
    cell = np.zeros((size, size), dtype=np.int32)
    
    # Central air hole: e.g. 8x8 square in the center
    # Indices 12 to 23 (8 pixels)
    start = size // 2 - 4
    end = size // 2 + 4
    cell[start:end, start:end] = 0
    
    # Now we need to optimize this.
    # We will use a local search.
    # To evaluate the score, we need to solve the eigenvalue problem.
    # This is the expensive part.
    
    # Simplified Solver for local feedback (approximate?)
    # No, we must be accurate.
    # 289 plane waves is 17x17.
    # Matrix is 289x289.
    # Eigen decomposition of 289x289 matrix is fast (~1ms).
    # 30 k-points -> 30 * 1ms = 30ms per evaluation.
    # 180s / 30ms = 6000 evaluations.
    # This is feasible!
    
    # We need to implement the plane wave solver.
    
    def compute_eps_fourier(cell):
        # cell is size x size array of 0s and 1s.
        # eps(r) = eps_low + (eps_high - eps_low) * cell(r)
        # We need eps_G for G = (mx * 2*pi, my * 2*pi) / size?
        # The period is 1 unit.
        # G vectors: mx, my in range [-N/2, N/2-1] for N=17?
        # 17 is odd. Range [-8, 8].
        N_pw = 17
        G_range = range(-N_pw//2, N_pw//2 + 1)
        
        # Precompute exp(-i G . r) for all pixels
        # r = (i+0.5)/size, (j+0.5)/size
        # We need integral over cell.
        # Integral is sum over pixels of value * area * exp(-i G . r_pixel)
        # Area = 1/size^2
        
        # Vectorized computation
        # G is a matrix of shape (N_G, 2)
        # Pixels are points.
        
        # Let's create the G vectors
        G_vectors = []
        for mx in G_range:
            for my in G_range:
                G_vectors.append((2*np.pi*mx, 2*np.pi*my))
        G_vectors = np.array(G_vectors)
        
        # Pixel centers
        i_coords, j_coords = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        i_coords = (i_coords + 0.5) / size
        j_coords = (j_coords + 0.5) / size
        
        # Flatten
        i_flat = i_coords.flatten()
        j_flat = j_coords.flatten()
        
        # Cell values
        cell_flat = cell.flatten()
        
        # Compute phase: -i * (G_x * x + G_y * y)
        # G_x is G_vectors[:, 0], x is i_flat
        # We need exp(-1j * (G_x[:, None] * i_flat[None, :] + G_y[:, None] * j_flat[None, :]))
        
        # This is (N_G, N_pixels)
        # N_G = 289, N_pixels = 1024
        # 289 * 1024 = ~300k complex exponentials. Fast.
        
        phase = -1j * (G_vectors[:, 0:1] * i_flat[None, :] + G_vectors[:, 1:2] * j_flat[None, :])
        exp_vals = np.exp(phase)
        
        # eps(r) = eps_low + (eps_high - eps_low) * cell_flat
        eps_vals = eps_low + (eps_high - eps_low) * cell_flat
        
        # Integral: sum(eps_vals * exp_vals) * (1/size^2)
        eps_fourier = np.sum(exp_vals * eps_vals[None, :], axis=1) / (size * size)
        
        return eps_fourier, G_vectors

    # We can cache eps_fourier for the current cell?
    # Since we flip pixels, we can update eps_fourier incrementally.
    # But for simplicity, let's recompute. 300k exp is very fast.
    
    # Actually, updating is better.
    # eps_G = sum_p eps_p exp(-i G . r_p) / A
    # If we flip pixel p from 0 to 1:
    # Delta eps = (eps_high - eps_low)
    # Delta eps_G = Delta eps * exp(-i G . r_p) / A
    
    # Let's implement the solver function.
    
    def solve_bands(cell):
        # Returns the score (gap-midgap ratio)
        
        eps_fourier, G_vectors = compute_eps_fourier(cell)
        
        # Map G vector index
        G_indices = {tuple(g): idx for idx, g in enumerate(G_vectors)}
        
        # Precompute G2 = |k+G|^2? No, k changes.
        
        # k-points in irreducible BZ
        # Triangle: (0,0) to (1,0) to (0.5, 0.5)
        # Wait, in units of 2pi/a?
        # If period is 1, reciprocal lattice vectors are 2pi.
        # k is in first BZ: k_x, k_y in [-pi, pi].
        # Irreducible sector: 0 <= k_y <= k_x <= pi.
        # Let's sample 30 points.
        
        # We can generate a grid and filter.
        # Or sample uniformly in the triangle.
        # Area of triangle: 0.5 * pi * pi = pi^2/2.
        
        # Let's just use a fixed set of k-points.
        # The task says "30 wave vectors". We should probably use the same ones as the evaluator if possible,
        # but we don't know them. Random uniform sampling in the triangle should be a good proxy.
        
        k_points = []
        # Gamma (0,0)
        k_points.append((0.0, 0.0))
        # M (pi, 0)
        k_points.append((np.pi, 0.0))
        # X (pi/2, pi/2)? No, X is (pi, 0) or (0, pi)?
        # For square lattice:
        # Gamma: (0,0)
        # X: (pi, 0)
        # M: (pi, pi)
        # Irreducible BZ is triangle Gamma-X-M?
        # No, Gamma(0,0), X(pi,0), M(pi,pi).
        # The sector is bounded by k_y=0, k_x=k_y, and k_x=pi?
        # Yes: 0 <= k_y <= k_x <= pi.
        
        # Generate random points in this triangle
        random.seed(42)
        for _ in range(29):
            u = random.random()
            v = random.random()
            # Transform to triangle
            # Triangle vertices: (0,0), (pi,0), (pi,pi)
            # Let x in [0, pi], y in [0, x]
            kx = pi * u
            ky = pi * u * v
            k_points.append((kx, ky))
            
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build Hamiltonian
            # H_mn = (k+G_m) . (k+G_n) * (1/eps_{G_m - G_n})
            
            # k is (kx, ky)
            # G_m is (Gx_m, Gy_m)
            # k+G_m is vector.
            
            # 1/eps_G array
            # eps_G is eps_fourier[G_index]
            # We need eps_{G_m - G_n}
            
            # Let's precompute inv_eps
            # Note: eps_0 is the average permittivity, which is non-zero.
            # Other eps_G might be small, but shouldn't be zero for a valid crystal.
            
            inv_eps = 1.0 / eps_fourier
            
            # Construct matrix
            # M = (k+G_m) . (k+G_n) * inv_eps[G_m - G_n]
            
            # G_diff_mn = G_m - G_n
            # We need to find the index in G_vectors for G_m - G_n.
            
            # Precompute a lookup table for G_m - G_n
            # G_vectors is 289x2.
            # G_m - G_n is a vector.
            
            # To speed this up, we can precompute the matrix of indices.
            # Let's do it once outside the k-loop?
            # G_vectors is fixed.
            
            # Create a dictionary for G vector -> index
            # We already have G_indices.
            
            # For each pair (m, n), find G_m - G_n.
            # This is 289*289 = 83k operations.
            # Doing this for every k-point (30 times) is 2.5 million operations.
            # This might be slow in Python loops.
            
            # Vectorize:
            # G_m is G_vectors[m]
            # G_n is G_vectors[n]
            # Diff = G_vectors[:, None, :] - G_vectors[None, :, :]
            # This is 289 x 289 x 2 array.
            # Flatten and look up?
            
            # Let's precompute the index matrix once.
            # Diff_vectors = G_vectors[:, None, :] - G_vectors[None, :, :]
            # Flatten to 1D? No, we need 2D lookup.
            
            # We can just loop in Python if we're careful, or use numpy tricks.
            # Let's try numpy tricks.
            
            # Create a list of all G differences
            # Actually, we can just precompute the full H matrix coefficients for each k.
            
            # Let's precompute the "structure factor" matrix S_mn = inv_eps[G_m - G_n]
            # S is 289x289.
            
            # To get S, we need to map each (G_m - G_n) to its index.
            
            # Let's create a 2D array of indices.
            idx_matrix = np.zeros((N_pw, N_pw, 2), dtype=np.int32) # m, n, dim
            # Wait, G_vectors is a list of tuples.
            
            # Let's make a dictionary for speed.
            # g_dict = {tuple(g): i for i, g in enumerate(G_vectors)}
            
            # For m in range(289):
            #   for n in range(289):
            #     diff = tuple(G_vectors[m] - G_vectors[n])
            #     idx = g_dict[diff]
            
            # This is 83k lookups. Fast enough?
            # 83k * 30 = 2.5M lookups. Might be ~0.5s.
            # We can do this once outside the k-loop.
            
            # Let's do it.
            
            # Precompute idx_matrix outside k-loop
            # This depends on G_vectors which is fixed.
            
            # Let's move this out.
            
            pass # Placeholder

        # Refactor: Precompute idx_matrix and S (structure factor)
        # S is constant for a given cell!
        # Only the (k+G_m).(k+G_n) part changes with k.
        
        # So:
        # 1. Compute eps_fourier
        # 2. Compute inv_eps
        # 3. Compute S_mn = inv_eps[G_m - G_n]
        # 4. For each k:
        #    a. Compute V_m = k + G_m
        #    b. H_mn = V_m . V_n * S_mn
        #    c. Eigenvalues of H
        #    d. Extract bands 4 and 5
        
        # This is much faster.
        
        # Recompute S
        # g_dict = {tuple(g): i for i, g in enumerate(G_vectors)}
        # S = np.zeros((N_pw, N_pw))
        # for m in range(N_pw):
        #   for n in range(N_pw):
        #     diff = tuple(G_vectors[m] - G_vectors[n])
        #     idx = g_dict[diff]
        #     S[m, n] = inv_eps[idx]
        
        # This loop is 83k iterations.
        # In Python, 83k iterations is ~10ms.
        # Acceptable.
        
        # Let's implement this.
        
        # Re-initialize G_vectors and eps_fourier
        # (Already done in compute_eps_fourier)
        
        # Compute S
        g_dict = {tuple(g): i for i, g in enumerate(G_vectors)}
        S = np.zeros((N_pw, N_pw))
        for m in range(N_pw):
            for n in range(N_pw):
                diff = tuple(G_vectors[m] - G_vectors[n])
                idx = g_dict[diff]
                S[m, n] = inv_eps[idx]
        
        # k-points
        # (Generated above)
        
        # Loop k-points
        for kx, ky in k_points:
            # V_m = k + G_m
            # V is 289x2
            V = G_vectors + np.array([kx, ky])
            
            # V_m . V_n = V[:, 0] * V[n, 0] + V[:, 1] * V[n, 1]
            # = V[:, 0, None] * V[None, :, 0] + V[:, 1, None] * V[None, :, 1]
            dot_prod = np.sum(V[:, None, :] * V[None, :, :], axis=2)
            
            H = dot_prod * S
            
            # Solve eigenvalues
            # H is symmetric
            eigvals = np.linalg.eigvalsh(H)
            
            # Sort eigenvalues
            # eigvalsh returns sorted in ascending order.
            
            # We want bands 4 and 5 (indices 3 and 4)
            # omega = sqrt(eigval)
            # But we can work with eigvals (omega^2)
            
            omega_sq_4 = eigvals[3]
            omega_sq_5 = eigvals[4]
            
            # We want min(omega_5) and max(omega_4)
            # omega = sqrt(omega_sq)
            # min(omega_5) = sqrt(min(omega_sq_5))
            # max(omega_4) = sqrt(max(omega_sq_4))
            
            # Ratio = 2 (min_w5 - max_w4) / (min_w5 + max_w4)
            # This is better computed directly on omegas to avoid sqrt errors?
            # Or we can compute on omega_sq?
            # Let w4 = sqrt(w4_sq), w5 = sqrt(w5_sq)
            
            w4 = np.sqrt(omega_sq_4)
            w5 = np.sqrt(omega_sq_5)
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == 0 or max_omega_4 == 0:
            return 0.0
            
        ratio = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return ratio

    # Main Optimization Loop
    
    best_score = solve_bands(cell)
    best_cell = cell.copy()
    
    current_cell = cell.copy()
    current_score = best_score
    
    T = 1.0 # Temperature
    T_min = 0.01
    
    # Track time
    start_time = time.time()
    
    # List of pixels to flip
    # We have `pixels` (fundamental domain) and `coverage`.
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose a random pixel to flip
        p_idx = random.randint(0, num_pixels - 1)
        i, j = pixels[p_idx]
        group = coverage[(i, j)]
        
        # Flip these pixels
        # Save old values
        old_vals = []
        for pi, pj in group:
            old_vals.append(current_cell[pi, pj])
            current_cell[pi, pj] = 1 - current_cell[pi, pj]
        
        # Evaluate
        new_score = solve_bands(current_cell)
        
        # Accept?
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = current_cell.copy()
        else:
            # Metropolis
            delta = new_score - current_score
            if random.random() < exp(delta / T):
                current_score = new_score
            else:
                # Revert
                for (pi, pj), val in zip(group, old_vals):
                    current_cell[pi, pj] = val
        
        # Cooldown
        if step % 50 == 0:
            T *= 0.95
            if T < T_min:
                T = T_min

    # Return best cell as nested list
    result = best_cell.tolist()
    return result

# EVOLVE-BLOCK-END
