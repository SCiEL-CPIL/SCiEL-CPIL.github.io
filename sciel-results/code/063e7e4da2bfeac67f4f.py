import numpy as np
import time
import math

def design(problem):
    """
    Designs a 32x32 symmetric photonic crystal unit cell by directly optimizing
    the binary pixel mask. Uses a C4v-symmetric orbit representation to reduce
    the search space and a fast local search (hill-climbing) with a lightweight
    band-structure evaluator.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # 1. Define the symmetry orbits for C4v about the center (15.5, 15.5)
    # A pixel (i, j) is equivalent to (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
    # and their transposes (j, i), etc.
    # We identify a unique representative for each orbit (the "leader") and store its value.
    
    visited = np.zeros((size, size), dtype=bool)
    orbits = []
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                # Find all pixels in this orbit
                orbit = []
                # Generate the 8 symmetries (some may be duplicates for center lines)
                coords = [
                    (i, j),
                    (size-1-i, j),
                    (i, size-1-j),
                    (size-1-i, size-1-j),
                    (j, i),
                    (j, size-1-i),
                    (size-1-j, i),
                    (size-1-j, size-1-i)
                ]
                # Unique coords
                unique_coords = []
                seen_coords = set()
                for c in coords:
                    if c not in seen_coords:
                        seen_coords.add(c)
                        unique_coords.append(c)
                
                for u in unique_coords:
                    visited[u[0], u[1]] = True
                    orbit.append(u)
                
                orbits.append(orbit)
    
    num_orbits = len(orbits)
    
    # 2. Fast Evaluator
    # We need to compute the TM band gap between bands 4 and 5.
    # The evaluator uses 289 plane waves (N=8) and 30 k-points.
    # We will replicate this as closely as possible but optimize for speed.
    
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Precompute k-points (30 points in irreducible BZ)
    # Gamma-M: (t, 0), M-X: (0.5, t), Gamma-X: (t, t)
    # Let's use a fixed set of 30 points.
    k_points = []
    # 10 points on Gamma-M
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0))
    # 10 points on M-X
    for t in np.linspace(0, 0.5, 10):
        k_points.append((0.5, t))
    # 10 points on Gamma-X
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, t))
    
    # Remove duplicates (Gamma, M, X are shared)
    unique_k = []
    seen_k = set()
    for k in k_points:
        if k not in seen_k:
            seen_k.add(k)
            unique_k.append(k)
    # Note: The problem says "30 wave vectors". My list might have duplicates at boundaries.
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5).
    # Gamma-M has (0,0) and (0.5,0). M-X has (0.5,0) and (0.5,0.5). Gamma-X has (0,0) and (0.5,0.5).
    # So unique count is 10 + 10 + 10 - 3 (duplicates) = 27? Or maybe the evaluator uses a specific grid.
    # To be safe, I will just use the 30 points I generated, handling duplicates by skipping if seen.
    # Actually, let's just use the 27 unique points or pad to 30 if needed?
    # The score depends on min/max over the specific k-points. If I miss one, I might miss the gap location.
    # But 27 points covering the paths is usually sufficient for a rough estimate.
    
    def evaluate_cell_binary(cell):
        """
        cell: 32x32 numpy array of 0/1.
        Returns the gap-midgap ratio.
        """
        # Compute epsilon grid
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        
        # Compute 1/epsilon
        inv_eps = 1.0 / eps_grid
        
        # Compute Fourier coefficients of 1/epsilon
        # The evaluator uses "exact Fourier coefficients of the pixel map".
        # This implies: eps_G = (1/A) * sum_{pixels} eps_p * exp(-i G . r_p)
        # For a 32x32 grid, the FFT gives the DFT coefficients.
        # DFT: F[k] = sum_{n} f[n] exp(-i 2 pi k n / N)
        # The physical Fourier coefficient is related to the DFT by normalization and scaling.
        # Specifically, if f(x) is a step function on [0,1]^2 with pixel width 1/32:
        # eps_G = (1/32^2) * sum_{i,j} eps_{i,j} exp(-i 2 pi (Gx i + Gy j) / 32)
        # This is exactly (1/N^2) * FFT(eps).
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        fft_inv_eps /= (size * size)
        
        # Extract coefficients for G in [-8, 8]
        # FFT indices 0..31 map to G = 0, 1, ..., 15, -16, ..., -1
        def get_coeff(gx, gy):
            ix = gx % size
            iy = gy % size
            return fft_inv_eps[ix, iy]
        
        # Precompute the matrix of Fourier coefficients for differences
        # We need (1/epsilon)_{G - G'}
        # G, G' are in [-8, 8]. So diff is in [-16, 16].
        # We can build a 33x33 array of coefficients centered at 0.
        # Or just use a dictionary or direct lookup.
        
        # For speed, let's precompute the "convolution kernel" matrix?
        # No, the matrix M depends on k.
        # M_{ij} = (1/eps)_{G_i - G_j} * |k + G_i|^2
        
        # Let's precompute the |k+G|^2 for each k and G?
        # That's 30 * 289 values. Small.
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            
            # Compute |k + G|^2 for all G
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G**2, axis=1)
            
            # Build the matrix M
            # M[i, j] = coeff[G_i - G_j] * omega_sq[i]
            # We can construct this using the FFT coefficients.
            # Let C be the 2D array of coefficients C[dx, dy] = (1/eps)_{dx, dy}
            # Then M[i, j] = C[G_i - G_j] * omega_sq[i]
            
            # To avoid a double loop, we can use the fact that M is a "banded" matrix
            # in the G-space, but the indices are not linear in G.
            # However, 289x289 is small. A double loop in Python is slow.
            # Let's try to vectorize the construction.
            
            # G_vectors is (289, 2).
            # We need a matrix Diff (289, 289, 2) where Diff[i, j] = G_i - G_j.
            # This is 289*289*2 floats ~ 160k floats. Manageable.
            
            # But we only need the coefficient for each diff.
            # Let's precompute a lookup table for coefficients for all possible diffs in [-16, 16].
            # coeff_table[dx+16, dy+16] = get_coeff(dx, dy)
            
            # This table is 33x33.
            # Then M[i, j] = coeff_table[Gx_i - Gx_j + 16, Gy_i - Gy_j + 16] * omega_sq[i]
            
            # Let's build the coeff_table once outside the k-loop?
            # Yes, it depends only on the cell.
            
            # I'll move coeff_table creation outside.
            
            # For now, inside the loop:
            # Create indices for the table
            # dx = Gx[:, None] - Gx[None, :]
            # dy = Gy[:, None] - Gy[None, :]
            # coeff_mat = coeff_table[dx + 16, dy + 16]
            # M = coeff_mat * omega_sq[:, None]
            
            # This is fast!
            
            # Wait, I need to define coeff_table outside.
            pass
        
        # I need to restructure to pass coeff_table.
        # Let's refactor evaluate_cell_binary to take coeff_table as input or compute it inside.
        # Computing it inside is fine (33x33 lookups).
        
        # Let's re-implement the loop with vectorized matrix construction.
        
        # Precompute Gx, Gy arrays (1D, length 289)
        Gx_flat = Gx.ravel()
        Gy_flat = Gy.ravel()
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G**2, axis=1)
            
            # Compute differences
            # dx[i, j] = Gx_flat[i] - Gx_flat[j]
            # dy[i, j] = Gy_flat[i] - Gy_flat[j]
            dx = Gx_flat[:, None] - Gx_flat[None, :]
            dy = Gy_flat[:, None] - Gy_flat[None, :]
            
            # Map to table indices
            # Table size 33x33, offset 16
            idx_x = dx + 16
            idx_y = dy + 16
            
            # Look up coefficients
            # We need to build coeff_table first.
            # Let's do it here for clarity, or pass it.
            # Building it every time is 33*33 = 1089 lookups. 30 times = 32k lookups. Negligible.
            
            coeff_mat = np.empty((num_pw, num_pw), dtype=complex)
            # We can use np.add.at or just a loop? No, vectorized lookup is best.
            # But coeff_table is a 2D array.
            # Let's build coeff_table once.
            
            # Actually, let's just build the matrix directly using the FFT array?
            # No, the FFT array is 32x32. The indices wrap around.
            # The table approach is cleaner.
            
            # Let's build coeff_table outside the k-loop.
            # I'll restructure the function.
            
            pass
            
        # Refactored version:
        
        # Build coeff_table (33x33)
        coeff_table = np.zeros((33, 33), dtype=complex)
        for dx in range(-16, 17):
            for dy in range(-16, 17):
                coeff_table[dx + 16, dy + 16] = get_coeff(dx, dy)
        
        # Now the k-loop
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G**2, axis=1)
            
            dx = Gx_flat[:, None] - Gx_flat[None, :]
            dy = Gy_flat[:, None] - Gy_flat[None, :]
            
            # Lookup
            coeff_mat = coeff_table[dx + 16, dy + 16]
            
            # M[i, j] = coeff_mat[i, j] * omega_sq[i]
            M = coeff_mat * omega_sq[:, None]
            
            # Solve eigenvalue problem
            # M is not Hermitian.
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # 3. Optimization
    # Start with a reasonable guess: a dielectric rod or a few dots.
    # Let's try a "four dots" pattern (common for large gaps).
    # Or a single large dot.
    # Let's start with a single central dot of radius ~ 0.2.
    
    cell = np.zeros((size, size), dtype=int)
    y, x = np.mgrid[0:size, 0:size]
    # Center is 15.5, 15.5
    dist = np.sqrt((x - 15.5)**2 + (y - 15.5)**2)
    cell[dist < 5] = 1  # Radius 5 pixels
    
    # Enforce symmetry (it should be by construction, but just in case)
    # The orbit representation ensures we only modify leaders.
    
    # Initialize orbit values from the cell
    orbit_vals = np.zeros(num_orbits)
    for idx, orbit in enumerate(orbits):
        # Value is 0 or 1
        # All pixels in the orbit have the same value in a symmetric cell
        # Take the value of the first pixel
        i, j = orbit[0]
        orbit_vals[idx] = cell[i, j]
    
    current_score = evaluate_cell_binary(cell)
    best_score = current_score
    best_orbit_vals = orbit_vals.copy()
    best_cell = cell.copy()
    
    # Local Search (Hill Climbing)
    # Try flipping single orbits (and maybe pairs).
    # Since we have ~200 orbits, single flips are 200 evaluations per step.
    # Each evaluation is ~0.1-0.2s?
    # 200 * 0.2 = 40s per step. Too slow.
    # We need to be smarter.
    # Maybe evaluate a random subset of flips?
    # Or use a coarser search first?
    
    # Let's estimate evaluation time.
    # 289x289 eigvals is fast (~10ms).
    # 30 k-points * 10ms = 300ms per evaluation.
    # 200 flips * 0.3s = 60s per full sweep.
    # We have 210s. So maybe 2-3 sweeps?
    # That's not enough for a good local search.
    
    # Alternative: Simulated Annealing or Random Restarts.
    # Or: Coarse-to-fine.
    # 1. Randomize 10 orbits, evaluate.
    # 2. Keep best.
    # 3. Repeat.
    
    # Let's try a random search with local refinement.
    # 1. Generate 10 random candidates.
    # 2. Evaluate.
    # 3. Take best, do local search (flip 1 orbit at a time, but only try 20 random flips per step).
    
    num_random_starts = 5
    for r in range(num_random_starts):
        if time.time() > deadline - 30:
            break
            
        # Randomize orbits
        # Bias towards "high contrast" or specific patterns?
        # Just random 0/1 for now.
        # Or start from the current best and flip some.
        if r == 0:
            # Start from the initial guess (rod)
            ov = orbit_vals.copy()
        else:
            # Random restart
            ov = np.random.randint(0, 2, num_orbits)
            # Maybe enforce some symmetry? No, orbits are already symmetric.
            # Maybe bias towards lower fill fraction?
            # Dielectric fraction ~ 0.3 is often good.
            # If random, it's 0.5.
            # Let's force some zeros.
            # Pick 30% of orbits to be 1, rest 0.
            num_ones = int(0.3 * num_orbits)
            ones_idx = np.random.choice(num_orbits, num_ones, replace=False)
            ov = np.zeros(num_orbits)
            ov[ones_idx] = 1
            
        # Reconstruct cell
        cell = np.zeros((size, size), dtype=int)
        for idx, val in enumerate(ov):
            for i, j in orbits[idx]:
                cell[i, j] = val
                
        score = evaluate_cell_binary(cell)
        
        if score > best_score:
            best_score = score
            best_orbit_vals = ov.copy()
            best_cell = cell.copy()
            
        # Local search on this candidate
        # Try flipping 1 orbit at a time.
        # To save time, only try a subset of flips or stop early.
        # Let's do 10 local search steps.
        for step in range(10):
            if time.time() > deadline - 30:
                break
                
            improved = False
            # Try flipping each orbit? Too slow.
            # Try flipping a random subset of 10 orbits?
            # Or just the most promising?
            # Let's try all orbits but break if time is tight.
            # Actually, 200 flips * 0.3s = 60s.
            # If we do this for 10 steps, it's 600s. Too slow.
            
            # We need to speed up evaluation or reduce flips.
            # Let's reduce k-points for the local search?
            # Use 10 k-points instead of 30.
            # This makes eval 3x faster.
            # 200 flips * 0.1s = 20s per sweep.
            # 10 sweeps = 200s. Still tight.
            
            # Let's use 5 k-points for the local search.
            # 200 flips * 0.05s = 10s per sweep.
            # 10 sweeps = 100s.
            
            # I need a fast evaluator for local search.
            
            def evaluate_cell_fast(cell):
                # Use 5 k-points
                fast_k_points = [
                    (0, 0),
                    (0.5, 0),
                    (0.5, 0.5),
                    (0.25, 0),
                    (0.5, 0.25)
                ]
                # ... same logic but with fast_k_points
                # I'll copy the logic here for speed (avoid function call overhead if possible, 
                # but Python function calls are cheap compared to numpy ops).
                
                eps_grid = np.where(cell == 1, eps_high, eps_low)
                inv_eps = 1.0 / eps_grid
                fft_inv_eps = np.fft.fft2(inv_eps)
                fft_inv_eps /= (size * size)
                
                def get_coeff(gx, gy):
                    ix = gx % size
                    iy = gy % size
                    return fft_inv_eps[ix, iy]
                
                coeff_table = np.zeros((33, 33), dtype=complex)
                for dx in range(-16, 17):
                    for dy in range(-16, 17):
                        coeff_table[dx + 16, dy + 16] = get_coeff(dx, dy)
                        
                min_omega_5 = np.inf
                max_omega_4 = -np.inf
                
                for kx, ky in fast_k_points:
                    k = np.array([kx, ky])
                    k_plus_G = k + G_vectors
                    omega_sq = np.sum(k_plus_G**2, axis=1)
                    
                    dx = Gx_flat[:, None] - Gx_flat[None, :]
                    dy = Gy_flat[:, None] - Gy_flat[None, :]
                    
                    coeff_mat = coeff_table[dx + 16, dy + 16]
                    M = coeff_mat * omega_sq[:, None]
                    
                    eigvals = np.linalg.eigvals(M)
                    real_eigvals = np.sort(np.real(eigvals))
                    
                    if len(real_eigvals) >= 5:
                        omega_4 = np.sqrt(real_eigvals[3])
                        omega_5 = np.sqrt(real_eigvals[4])
                        
                        if omega_4 > max_omega_4:
                            max_omega_4 = omega_4
                        if omega_5 < min_omega_5:
                            min_omega_5 = omega_5
                
                if min_omega_5 == np.inf or max_omega_4 == -np.inf:
                    return -1.0
                gap = min_omega_5 - max_omega_4
                midgap = 0.5 * (min_omega_5 + max_omega_4)
                if midgap == 0:
                    return -1.0
                return 2 * gap / midgap

            current_fast_score = evaluate_cell_fast(cell)
            
            # Try flipping orbits
            # To make it faster, I'll shuffle the orbits and try flipping them one by one,
            # accepting the first improvement (greedy).
            # If no improvement after checking all, stop.
            
            orbit_indices = np.random.permutation(num_orbits)
            found_improvement = False
            
            for idx in orbit_indices:
                if time.time() > deadline - 30:
                    break
                    
                # Flip
                ov[idx] = 1 - ov[idx]
                
                # Reconstruct cell (only the changed pixels)
                # Instead of full reconstruction, I can update the cell in place?
                # No, evaluate_cell_fast expects a full cell.
                # But I can create a temporary cell.
                # To save time, I'll just reconstruct the whole cell.
                # 32x32 is small.
                
                new_cell = np.zeros((size, size), dtype=int)
                for o_idx, val in enumerate(ov):
                    for i, j in orbits[o_idx]:
                        new_cell[i, j] = val
                        
                new_score = evaluate_cell_fast(new_cell)
                
                if new_score > current_fast_score:
                    # Accept
                    cell = new_cell
                    current_fast_score = new_score
                    found_improvement = True
                    # Don't break, keep looking for more improvements?
                    # Or break and restart the loop?
                    # Let's keep looking.
                else:
                    # Revert
                    ov[idx] = 1 - ov[idx]
                    
            if not found_improvement:
                break
                
        # After local search, evaluate the best candidate with the full evaluator
        final_score = evaluate_cell_binary(cell)
        if final_score > best_score:
            best_score = final_score
            best_orbit_vals = ov.copy()
            best_cell = cell.copy()

    # Return the best cell found
    # Ensure it's a nested list of ints
    return best_cell.tolist()

# Test/Debug
if __name__ == "__main__":
    problem = {
        "pixels": 32,
        "eps_low": 1.0,
        "eps_high": 12.0,
        "lower_band": 4,
        "polarization": "TM"
    }
    cell = design(problem)
    print("Cell shape:", len(cell), "x", len(cell[0]))
    print("Sum of dielectric pixels:", sum(sum(row) for row in cell))
