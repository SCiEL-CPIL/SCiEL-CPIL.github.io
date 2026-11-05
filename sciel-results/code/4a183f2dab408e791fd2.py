import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 200  # 20s margin from 220s limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Pre-compute symmetry orbits for the 32x32 grid
    # Symmetry group: C4v (square symmetry about center)
    # Center is at (15.5, 15.5)
    # Orbits are sets of pixels that must have the same value.
    
    orbits = []
    visited = np.zeros((size, size), dtype=bool)
    
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                orbit = []
                # Generate all symmetric equivalents of (i, j)
                # 1. Identity: (i, j)
                # 2. Rot 90: (j, size-1-i)
                # 3. Rot 180: (size-1-i, size-1-j)
                # 4. Rot 270: (size-1-j, i)
                # 5. Refl X (vertical axis through center): (i, size-1-j)
                # 6. Refl Y (horizontal axis through center): (size-1-i, j)
                # 7. Refl Diag 1 (i=j): (j, i)
                # 8. Refl Diag 2 (i+j=size-1): (size-1-j, size-1-i)
                
                candidates = [
                    (i, j),
                    (j, size-1-i),
                    (size-1-i, size-1-j),
                    (size-1-j, i),
                    (i, size-1-j),
                    (size-1-i, j),
                    (j, i),
                    (size-1-j, size-1-i)
                ]
                
                for r, c in candidates:
                    if 0 <= r < size and 0 <= c < size:
                        orbit.append((r, c))
                        visited[r, c] = True
                
                # Unique orbit members
                unique_orbit = list(set(orbit))
                orbits.append(unique_orbit)
    
    num_orbits = len(orbits)
    
    # Fast PWE Solver
    # We need to solve for TM bands.
    # H = A * H
    # A is the matrix of 1/eps(k-G, k-G')
    # For TM polarization, the operator is (1/eps) * (k-G)^2
    # Wait, standard PWE for TM:
    # (k-G) . (k-G') / eps(k-G-k-G') ... no.
    # The eigenvalue problem is:
    # sum_{G'} (1/eps(G-G')) (k-G) . (k-G') A_{G'} = omega^2 A_G
    # Let's use the standard formulation:
    # M_{GG'} = (k-G) . (k-G') / eps(k-G - k-G')
    # This is for TE?
    # For TM:
    # The vector potential A is scalar (z-component).
    # Maxwell: curl curl A = omega^2 / c^2 * eps * A
    # In Fourier: (k-G)^2 A_G - sum_{G'} (1/eps_{G-G'}) (k-G) . (k-G') A_{G'} = 0 ?
    # Actually, the standard matrix form for TM is:
    # H_{GG'} = (k-G) . (k-G') / eps_{G-G'}
    # Where eps_{G} is the Fourier coefficient of 1/eps(r).
    
    # Precompute Fourier coefficients of 1/eps for the given pixel map
    # We will do this inside the evaluation function for speed, but optimized.
    
    # Grid of G-vectors
    # We need a set of G-vectors that covers the reciprocal space sufficiently.
    # Size 32. Reciprocal lattice vectors are integer multiples of 2pi/size.
    # Let's use a square grid of G-vectors up to a certain order N.
    # N=6 or 7 is usually sufficient for 32x32 with eps=12.
    N = 7
    G_vectors = []
    for gx in range(-N, N+1):
        for gy in range(-N, N+1):
            G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors)
    num_G = len(G_vectors)
    
    # K-points in irreducible BZ
    # Square lattice BZ: -pi/a to pi/a. Irreducible: 0 <= ky <= kx <= pi/a
    # Let's sample a grid of k-points.
    k_samples = []
    num_k = 10
    for ix in range(num_k):
        for iy in range(num_k):
            kx = ix / (num_k - 1) * math.pi
            ky = iy / (num_k - 1) * math.pi
            if ky <= kx: # Irreducible sector
                k_samples.append((kx, ky))
    k_samples = np.array(k_samples)
    
    def compute_eps_fourier(pixel_grid):
        """
        Compute Fourier coefficients of 1/eps(r).
        pixel_grid: 32x32 array of 0/1.
        1 = dielectric (eps_high), 0 = air (eps_low).
        """
        # 1/eps values
        inv_eps_grid = np.zeros((size, size))
        inv_eps_grid[pixel_grid == 1] = 1.0 / eps_high
        inv_eps_grid[pixel_grid == 0] = 1.0 / eps_low
        
        # FFT
        # We need coefficients for G = (gx, gy)
        # eps_G = 1/(size^2) * sum_{i,j} eps_{ij} exp(-i G . r_ij)
        # Using FFT:
        # Shift grid so origin is at center? 
        # Standard FFT assumes origin at (0,0).
        # Our pixel (i,j) is at position (i, j) in units of pixel size.
        # G . r = (gx * 2pi/size * i) + (gy * 2pi/size * j)
        # This matches the standard DFT definition if we treat i,j as indices.
        
        fft_grid = np.fft.fft2(inv_eps_grid)
        # Normalize by size^2
        fft_grid /= (size * size)
        
        # Extract coefficients for our G_vectors
        coeffs = np.zeros(num_G, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            # Map gx, gy to FFT indices
            # FFT index for frequency k is k % size
            ix = gx % size
            iy = gy % size
            coeffs[idx] = fft_grid[iy, ix]
            
        return coeffs
    
    def evaluate_gap(pixel_grid):
        """
        Calculate the gap between band 4 and 5.
        Returns the gap-midgap ratio.
        """
        if time.time() > deadline:
            return -1.0 # Timeout indicator
        
        eps_coeffs = compute_eps_fourier(pixel_grid)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_samples:
            # Construct H matrix
            # H_{GG'} = (k-G) . (k-G') / eps_{G-G'}
            # k is in units of 2pi/size? 
            # Let's work in units where a=1. k is in radians/length.
            # G_vectors are integers. Actual G = G_vec * 2pi/size.
            # Let's scale k and G to be in units of 2pi/size.
            # k_scaled = k * size / (2pi)
            # G_scaled = G_vec
            # Then (k-G) is in scaled units.
            # The eigenvalue omega^2 will be in scaled units too.
            # We just need the ratio, so scaling cancels out.
            
            k_scaled = np.array([kx * size / (2*math.pi), ky * size / (2*math.pi)])
            
            # Vector of k-G for all G
            # G_vectors is (num_G, 2)
            # k_scaled is (2,)
            KG = G_vectors - k_scaled  # (num_G, 2)
            
            # Dot products (k-G) . (k-G')
            # KG @ KG.T gives (num_G, num_G) matrix of dot products
            dot_products = KG @ KG.T
            
            # eps_{G-G'}
            # We need eps coefficients for differences G - G'
            # This is a convolution.
            # eps_{G-G'} = sum_{H} eps_H eps_{G-G'-H} ? No, that's for product.
            # We have the Fourier coefficients of 1/eps directly.
            # Let's call them c_H where H is a G-vector.
            # We need c_{G-G'} for all pairs G, G'.
            
            # Optimization: Precompute the matrix of indices for G-G'
            # Or just compute it on the fly if num_G is small.
            # num_G ~ (2*7+1)^2 = 225. 225x225 = 50k entries.
            # Doing this for every k-point is expensive.
            
            # Better approach:
            # The matrix M is M_{GG'} = dot(GG') * c_{G-G'}
            # We can construct the matrix of c_{G-G'} once for a given pixel_grid?
            # No, c depends on pixel_grid.
            
            # Let's build the c_diff matrix
            # c_diff[i, j] = c[G_i - G_j]
            # We need to map vector difference to index in eps_coeffs.
            
            # Create a lookup table for G-vector differences
            # Since G_vectors are in a range [-N, N], differences are in [-2N, 2N].
            # We can create a dictionary or use modulo indexing if we extend the FFT.
            
            # Simpler: Just loop. 225*225 = 50625 operations.
            # 30 k-points * 50625 = 1.5M operations per evaluation.
            # Python loops are slow. We need to vectorize or use NumPy.
            
            # Let's precompute the indices of G-G' in the extended FFT grid?
            # Or just compute the matrix using broadcasting?
            
            # Let's assume we can compute c_diff matrix efficiently.
            # For now, let's use a sparse approach or just accept the cost if we reduce N.
            # Let's reduce N to 5 to speed up.
            # N=5 -> 11x11 = 121 G-vectors. 121^2 = 14641. Much better.
            
            # Re-evaluate N choice.
            # With N=5, resolution might be slightly lower, but should be okay for ranking.
            
            # Let's stick with N=5 for speed.
            # If time permits, we can refine with N=6 or 7 for the final candidate.
            
            # Constructing c_diff matrix
            # We need c_{G_i - G_j}
            # Let's create an array of all differences
            # G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
            # This is (num_G, num_G, 2)
            
            # We need to look up c for each difference.
            # c is defined for G in [-N, N]. Differences are in [-2N, 2N].
            # We need to compute the FFT for the larger range?
            # Or just compute the Fourier coefficient directly for the difference?
            # c_H = 1/S sum exp(-i H r) 1/eps(r)
            # This is expensive to do for every pair.
            
            # Alternative: Use the property that c_{G-G'} is the correlation of the grid.
            # Actually, let's just compute the matrix M directly using the grid?
            # No, PWE requires the matrix form.
            
            # Let's optimize the lookup.
            # Precompute a map from (gx, gy) to index in eps_coeffs.
            # But differences might be outside [-N, N].
            # We should compute eps_coeffs for a larger range, say [-2N, 2N].
            # Let's re-compute eps_coeffs with N_diff = 2*N.
            
            # Let's refactor: compute_eps_fourier returns coeffs for range [-N_max, N_max].
            # We will use N_max = 2*N for the differences.
            
            # Let's restart the solver logic with this in mind.
            
            pass # Placeholder, logic will be implemented below
            
        return (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4) * 2.0

    # Refined Solver Implementation
    
    # We will use N=5 for the G-vectors in the eigenvalue problem.
    # We will compute eps_coeffs for differences up to 2*N.
    
    N_G = 5
    N_diff = 2 * N_G
    
    G_vectors = []
    for gx in range(-N_G, N_G+1):
        for gy in range(-N_G, N_G+1):
            G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors)
    num_G = len(G_vectors)
    
    # Precompute G_diff indices
    # We need to map (dx, dy) to an index in the eps_coeffs array computed with N_diff.
    # eps_coeffs array will be indexed by (gx + N_diff, gy + N_diff)
    
    # Create a lookup table for differences
    # diff_indices[i, j] = index in eps_coeffs for G_i - G_j
    diff_indices = np.zeros((num_G, num_G), dtype=int)
    for i in range(num_G):
        for j in range(num_G):
            dx = G_vectors[i, 0] - G_vectors[j, 0]
            dy = G_vectors[i, 1] - G_vectors[j, 1]
            # Map to 0..2*N_diff
            ix = dx + N_diff
            iy = dy + N_diff
            diff_indices[i, j] = iy * (2*N_diff + 1) + ix
            
    # Precompute dot products for all k-points?
    # No, k-points are few, but G-vectors are many.
    # We can precompute KG for each k-point.
    
    def fast_evaluate(pixel_grid):
        """
        Fast evaluation of the gap.
        """
        if time.time() > deadline:
            return -1.0
        
        # Compute 1/eps grid
        inv_eps_grid = np.zeros((size, size))
        inv_eps_grid[pixel_grid == 1] = 1.0 / eps_high
        inv_eps_grid[pixel_grid == 0] = 1.0 / eps_low
        
        # Compute FFT for larger range
        # We need coefficients for G in [-N_diff, N_diff]
        # We can use the FFT of the grid and extract the needed coefficients.
        # The FFT of a size x size grid gives frequencies 0..size-1.
        # We need frequencies -N_diff..N_diff.
        # Since size=32 and N_diff=10, range is -10..10.
        # FFT indices: 0..31.
        # Frequency k corresponds to index k if k >= 0, and k+32 if k < 0.
        
        fft_grid = np.fft.fft2(inv_eps_grid) / (size * size)
        
        # Extract coefficients into a flat array for easy indexing
        # We will create an array eps_coeffs_flat of size (2*N_diff+1)^2
        eps_coeffs_flat = np.zeros((2*N_diff + 1) * (2*N_diff + 1), dtype=complex)
        
        for gx in range(-N_diff, N_diff+1):
            for gy in range(-N_diff, N_diff+1):
                ix = gx % size
                iy = gy % size
                idx = (gy + N_diff) * (2*N_diff + 1) + (gx + N_diff)
                eps_coeffs_flat[idx] = fft_grid[iy, ix]
        
        # Now evaluate for each k-point
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_samples:
            k_scaled = np.array([kx * size / (2*math.pi), ky * size / (2*math.pi)])
            KG = G_vectors - k_scaled
            
            # Dot products
            dot_products = KG @ KG.T
            
            # Get eps_diff matrix
            # eps_diff[i, j] = eps_coeffs_flat[diff_indices[i, j]]
            eps_diff = eps_coeffs_flat[diff_indices]
            
            # H matrix
            H = dot_products * eps_diff
            
            # Solve eigenvalue problem
            # H is Hermitian? 
            # dot_products is real symmetric.
            # eps_diff is complex Hermitian (since eps(r) is real).
            # So H is Hermitian.
            try:
                eigvals = np.linalg.eigvalsh(H)
            except np.linalg.LinAlgError:
                continue
                
            # We want bands 4 and 5 (index 3 and 4, 0-based)
            # Note: The lowest bands might be spurious or high frequency?
            # In PWE, we usually look at the lowest bands.
            # Band 1 is the lowest.
            # We need the 4th and 5th lowest bands.
            
            if len(eigvals) < 5:
                continue
                
            omega_4 = np.sqrt(eigvals[3])
            omega_5 = np.sqrt(eigvals[4])
            
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
            
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return -1.0
            
        return (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4) * 2.0

    # Initialization
    # Start with a pattern that has some structure.
    # Let's try a "holey" square lattice.
    # Period 8, hole size 4.
    best_grid = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            # Center of each 8x8 block
            if (i % 8) < 4 and (j % 8) < 4:
                best_grid[i, j] = 1
            else:
                best_grid[i, j] = 0
                
    best_score = fast_evaluate(best_grid)
    
    # Local Search
    # We will flip orbits.
    # Sort orbits by some heuristic? Or random order?
    # Random order with restarts.
    
    current_grid = best_grid.copy()
    current_score = best_score
    
    # List of orbit indices
    orbit_indices = list(range(num_orbits))
    
    steps = 0
    max_steps = 5000 # Limit steps to ensure we finish
    
    while steps < max_steps and time.time() < deadline - 10:
        np.random.shuffle(orbit_indices)
        improved = False
        
        for o_idx in orbit_indices:
            # Flip the orbit
            orbit = orbits[o_idx]
            # Store original values
            orig_vals = [current_grid[r, c] for r, c in orbit]
            
            # Flip
            for r, c in orbit:
                current_grid[r, c] = 1 - current_grid[r, c]
                
            new_score = fast_evaluate(current_grid)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_grid = current_grid.copy()
            else:
                # Revert
                for k, (r, c) in enumerate(orbit):
                    current_grid[r, c] = orig_vals[k]
                    
        if not improved:
            # Kick: Randomly flip a few orbits to escape local minimum
            for _ in range(5):
                o_idx = np.random.choice(num_orbits)
                orbit = orbits[o_idx]
                for r, c in orbit:
                    current_grid[r, c] = 1 - current_grid[r, c]
                    
            # Accept kick if it doesn't ruin the score too much? 
            # Or just accept it and continue searching.
            # Let's accept it to explore.
            current_score = fast_evaluate(current_grid)
            
        steps += 1
        
    return best_grid.tolist()

# Main entry point
def design(problem):
    start_time = time.time()
    deadline = start_time + 200
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # ... (Copy the solver code from above) ...
    
    # For the sake of the complete program, I will re-structure it cleanly.
    
    # Symmetry Orbits
    orbits = []
    visited = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                orbit = []
                candidates = [
                    (i, j),
                    (j, size-1-i),
                    (size-1-i, size-1-j),
                    (size-1-j, i),
                    (i, size-1-j),
                    (size-1-i, j),
                    (j, i),
                    (size-1-j, size-1-i)
                ]
                for r, c in candidates:
                    if 0 <= r < size and 0 <= c < size:
                        orbit.append((r, c))
                        visited[r, c] = True
                orbits.append(list(set(orbit)))
                
    num_orbits = len(orbits)
    
    # PWE Setup
    N_G = 5
    N_diff = 2 * N_G
    
    G_vectors = []
    for gx in range(-N_G, N_G+1):
        for gy in range(-N_G, N_G+1):
            G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors)
    num_G = len(G_vectors)
    
    diff_indices = np.zeros((num_G, num_G), dtype=int)
    for i in range(num_G):
        for j in range(num_G):
            dx = G_vectors[i, 0] - G_vectors[j, 0]
            dy = G_vectors[i, 1] - G_vectors[j, 1]
            ix = dx + N_diff
            iy = dy + N_diff
            diff_indices[i, j] = iy * (2*N_diff + 1) + ix
            
    k_samples = []
    num_k = 10
    for ix in range(num_k):
        for iy in range(num_k):
            kx = ix / (num_k - 1) * math.pi
            ky = iy / (num_k - 1) * math.pi
            if ky <= kx:
                k_samples.append((kx, ky))
    k_samples = np.array(k_samples)
    
    def fast_evaluate(pixel_grid):
        if time.time() > deadline:
            return -1.0
        
        inv_eps_grid = np.zeros((size, size))
        inv_eps_grid[pixel_grid == 1] = 1.0 / eps_high
        inv_eps_grid[pixel_grid == 0] = 1.0 / eps_low
        
        fft_grid = np.fft.fft2(inv_eps_grid) / (size * size)
        
        eps_coeffs_flat = np.zeros((2*N_diff + 1) * (2*N_diff + 1), dtype=complex)
        for gx in range(-N_diff, N_diff+1):
            for gy in range(-N_diff, N_diff+1):
                ix = gx % size
                iy = gy % size
                idx = (gy + N_diff) * (2*N_diff + 1) + (gx + N_diff)
                eps_coeffs_flat[idx] = fft_grid[iy, ix]
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_samples:
            k_scaled = np.array([kx * size / (2*math.pi), ky * size / (2*math.pi)])
            KG = G_vectors - k_scaled
            dot_products = KG @ KG.T
            eps_diff = eps_coeffs_flat[diff_indices]
            H = dot_products * eps_diff
            
            try:
                eigvals = np.linalg.eigvalsh(H)
            except np.linalg.LinAlgError:
                continue
                
            if len(eigvals) < 5:
                continue
                
            omega_4 = np.sqrt(eigvals[3])
            omega_5 = np.sqrt(eigvals[4])
            
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
            
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return -1.0
            
        return (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4) * 2.0

    # Initialization: Square lattice of holes
    best_grid = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if (i % 8) < 4 and (j % 8) < 4:
                best_grid[i, j] = 1
            else:
                best_grid[i, j] = 0
                
    best_score = fast_evaluate(best_grid)
    current_grid = best_grid.copy()
    current_score = best_score
    
    orbit_indices = list(range(num_orbits))
    steps = 0
    max_steps = 5000
    
    while steps < max_steps and time.time() < deadline - 10:
        np.random.shuffle(orbit_indices)
        improved = False
        
        for o_idx in orbit_indices:
            orbit = orbits[o_idx]
            orig_vals = [current_grid[r, c] for r, c in orbit]
            
            for r, c in orbit:
                current_grid[r, c] = 1 - current_grid[r, c]
                
            new_score = fast_evaluate(current_grid)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_grid = current_grid.copy()
            else:
                for k, (r, c) in enumerate(orbit):
                    current_grid[r, c] = orig_vals[k]
                    
        if not improved:
            for _ in range(5):
                o_idx = np.random.choice(num_orbits)
                orbit = orbits[o_idx]
                for r, c in orbit:
                    current_grid[r, c] = 1 - current_grid[r, c]
            current_score = fast_evaluate(current_grid)
            
        steps += 1
        
    return best_grid.tolist()
