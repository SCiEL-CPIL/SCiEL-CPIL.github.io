import time
import numpy as np
import math

def design(problem):
    """
    Optimizes a 32x32 photonic crystal unit cell for the TM band gap between bands 4 and 5.
    Uses a fast plane-wave approximation for local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    # 1. Initialize with a square rod, which often performs better than a circle for higher gaps
    # Side length approx 0.45
    side = 0.45
    cell = np.zeros((size, size), dtype=int)
    center = size / 2.0
    half_side = side / 2.0
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < half_side and abs(y) < half_side:
                cell[i, j] = 1
                
    # Ensure D4 symmetry explicitly (though the square init is symmetric)
    # We will operate on the fundamental domain to enforce symmetry during search
    # Fundamental domain: 0 <= i < size/2, 0 <= j < size/2, and i <= j is not strictly required 
    # if we just replicate the top-left quadrant to all 4 quadrants with reflections.
    # Square symmetry: 
    # Q1 (top-left, i<16, j<16)
    # Q2 (top-right, i<16, j>=16) is mirror of Q1
    # Q3 (bottom-right) is 180 rot of Q1
    # Q4 (bottom-left) is mirror of Q2
    
    def enforce_symmetry(c):
        """Returns a new array with D4 symmetry enforced from the top-left quadrant."""
        half = size // 2
        q1 = c[:half, :half].copy()
        # Mirror horizontally for top-right
        q2 = q1[:, ::-1]
        # Mirror vertically for bottom-left
        q4 = q1[::-1, :]
        # 180 rotation for bottom-right
        q3 = q1[::-1, ::-1]
        
        top = np.hstack((q1, q2))
        bottom = np.hstack((q4, q3))
        return np.vstack((top, bottom))

    # Apply initial symmetry
    cell = enforce_symmetry(cell)

    # 2. Fast Plane-Wave Solver
    # We use a smaller basis (e.g., 11x11 = 121 PWs) for speed during search.
    # The evaluator uses 289 (17x17). 121 is a good trade-off.
    N_PW = 11
    # Reciprocal lattice vectors for square lattice (a=1)
    # G = (h, k) where h, k are integers
    # We need a square patch of G-vectors centered at 0
    G_vectors = []
    for h in range(-N_PW // 2, N_PW // 2 + 1):
        for k in range(-N_PW // 2, N_PW // 2 + 1):
            G_vectors.append((h, k))
    N_G = len(G_vectors)
    
    # Precompute Fourier coefficients of epsilon for the current cell
    # eps(x,y) = eps_low + (eps_high - eps_low) * pixel(x,y)
    # We need the Fourier coefficients of the pixel function P(x,y)
    # P(x,y) is 1 if dielectric, 0 if air.
    
    def compute_fourier_coeffs(pixel_grid):
        """
        Computes the Fourier coefficients of the pixel grid.
        pixel_grid is size x size.
        Returns a dict {(h,k): coeff} or a 2D array indexed by G_vectors.
        """
        coeffs = np.zeros(N_G, dtype=complex)
        # The function is periodic on [0,1]x[0,1]
        # c_hk = integral_0^1 integral_0^1 P(x,y) exp(-i 2pi (hx + ky)) dx dy
        # Approximate integral by sum over pixels
        # Pixel (i,j) covers [i/size, (i+1)/size] x [j/size, (j+1)/size]
        # Center is (i+0.5)/size, (j+0.5)/size
        # Value is 1 if pixel is 1, 0 otherwise.
        
        # Optimization: Only sum over pixels that are 1
        # But we need to do this fast.
        # Let's use a direct sum. N_G is ~121. Size is 32. 121*32*32 is ~124k ops. Fast enough.
        
        for idx, (h, k) in enumerate(G_vectors):
            if h == 0 and k == 0:
                # Average value
                coeffs[idx] = np.sum(pixel_grid) / (size * size)
                continue
            
            # Sum over pixels
            # exp(-i 2pi (h*x + k*y))
            # x = (i+0.5)/size, y = (j+0.5)/size
            # We can precompute the exponentials for each h,k
            # But let's just do it inline for simplicity, it's fast.
            
            sum_val = 0.0 + 0.0j
            # Iterate over pixels where pixel_grid[i,j] == 1
            # To be faster, we can precompute the list of active pixels
            # But for now, direct loop.
            for i in range(size):
                for j in range(size):
                    if pixel_grid[i, j] == 1:
                        x = (i + 0.5) / size
                        y = (j + 0.5) / size
                        angle = -2.0 * math.pi * (h * x + k * y)
                        sum_val += math.cos(angle) + 1j * math.sin(angle)
            
            coeffs[idx] = sum_val / (size * size)
            
        return coeffs

    # Precompute pixel coordinates for speed
    pixel_coords = np.zeros((size, size, 2))
    for i in range(size):
        for j in range(size):
            pixel_coords[i, j, 0] = (i + 0.5) / size
            pixel_coords[i, j, 1] = (j + 0.5) / size

    def fast_gap_calculation(pixel_grid):
        """
        Calculates the TM band gap ratio between bands 4 and 5 using a simplified PW method.
        Returns the gap-midgap ratio.
        """
        # 1. Get Fourier coefficients of 1/epsilon? 
        # For TM, the equation is:
        # sum_k' [ (G-G')^2 / eps(G-G') ] E_k' = (omega^2 / c^2) E_k
        # Wait, for TM, the field is H_z. The equation involves 1/epsilon.
        # Let's check the standard formulation.
        # TM: H_z satisfies:
        # sum_{G'} (k+G)^2 / epsilon(G-G') H_{G'} = (omega/c)^2 H_k
        # Wait, usually it's:
        # sum_{G'} (G-G')^2 / epsilon(G-G') ... no.
        # Standard PW for TM:
        # The operator is (1/eps) * (k+G)^2.
        # So we need the Fourier coefficients of 1/epsilon.
        
        # Let's compute 1/epsilon pixel map
        inv_eps = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if pixel_grid[i, j] == 1:
                    inv_eps[i, j] = 1.0 / eps_high
                else:
                    inv_eps[i, j] = 1.0 / eps_low
        
        # Compute Fourier coeffs of 1/epsilon
        # f_hk = sum pixels (1/eps)_ij * exp(-i 2pi (hx+ky)) / N_pixels
        # We can reuse the logic but with values from inv_eps.
        
        coeffs_inv_eps = np.zeros(N_G, dtype=complex)
        for idx, (h, k) in enumerate(G_vectors):
            if h == 0 and k == 0:
                coeffs_inv_eps[idx] = np.mean(inv_eps)
                continue
            
            sum_val = 0.0 + 0.0j
            # Vectorized sum for speed
            # x = pixel_coords[:, :, 0], y = pixel_coords[:, :, 1]
            # angle = -2pi (h*x + k*y)
            # weights = inv_eps
            # sum_val = np.sum(weights * np.exp(1j * angle))
            
            # To avoid creating large temp arrays, we can do it in chunks or just use numpy
            x = pixel_coords[:, :, 0]
            y = pixel_coords[:, :, 1]
            angles = -2.0 * math.pi * (h * x + k * y)
            exp_vals = np.exp(1j * angles)
            sum_val = np.sum(inv_eps * exp_vals)
            
            coeffs_inv_eps[idx] = sum_val / (size * size)

        # 2. Construct the matrix A
        # A_{k, k'} = (k+G_k)^2 / epsilon(G_k - G_k')
        # Wait, the index G_k - G_k' corresponds to the difference vector.
        # We need to map difference vectors to our G_vectors list.
        
        # Build a lookup for G_vectors
        G_lookup = {}
        for idx, (h, k) in enumerate(G_vectors):
            G_lookup[(h, k)] = idx
            
        # Matrix A
        A = np.zeros((N_G, N_G), dtype=complex)
        
        for i_idx, (hi, ki) in enumerate(G_vectors):
            # (k+G_i)^2 is the magnitude squared of the wave vector in the unit cell
            # k is the Bloch wave vector. We evaluate at a specific k.
            # For gap calculation, we need to scan k in the BZ.
            # This function takes a specific k.
            pass # Placeholder, we will implement the k-scan inside the search loop or return a function.

    # This approach is getting complex to implement fully in a single function with time limits.
    # Let's simplify: We will scan a few key k-points (Gamma, X, M, and a few in between) 
    # and compute the eigenvalues.
    
    def compute_bands_at_k(kx, ky, pixel_grid):
        """
        Computes the first 5 TM bands at a given k-point (kx, ky) in units of 2pi/a.
        """
        # 1. Fourier coeffs of 1/epsilon
        inv_eps = np.where(pixel_grid == 1, 1.0 / eps_high, 1.0 / eps_low)
        
        coeffs = np.zeros(N_G, dtype=complex)
        for idx, (h, k) in enumerate(G_vectors):
            if h == 0 and k == 0:
                coeffs[idx] = np.mean(inv_eps)
            else:
                x = pixel_coords[:, :, 0]
                y = pixel_coords[:, :, 1]
                angles = -2.0 * math.pi * (h * x + k * y)
                coeffs[idx] = np.sum(inv_eps * np.exp(1j * angles)) / (size * size)
                
        # 2. Build Matrix
        # A_{mn} = (k + G_m)^2 / eps(G_m - G_n)
        # Note: G_m - G_n must be in our set of G_vectors.
        # If N_PW is 11, the differences range from -10 to 10.
        # Our G_vectors cover -5 to 5. So differences might be outside!
        # We need a larger set of G vectors for the *difference* or ensure our basis is large enough.
        # If we use N_PW=11, the G vectors are -5..5.
        # Difference of two vectors in -5..5 is -10..10.
        # So we need coeffs for -10..10.
        # This means our Fourier coefficient calculation must cover a larger range, 
        # or we must use a larger basis for the matrix construction.
        
        # Let's increase N_PW to 15 for the solver to be safe with differences.
        # 15x15 = 225 PWs. Eigenvalue decomposition of 225x225 is fast.
        
        N_PW_solver = 15
        G_vectors_solver = []
        for h in range(-N_PW_solver // 2, N_PW_solver // 2 + 1):
            for k in range(-N_PW_solver // 2, N_PW_solver // 2 + 1):
                G_vectors_solver.append((h, k))
        N_G_solver = len(G_vectors_solver)
        
        # Recompute coeffs for larger basis
        coeffs_solver = np.zeros(N_G_solver, dtype=complex)
        for idx, (h, k) in enumerate(G_vectors_solver):
            if h == 0 and k == 0:
                coeffs_solver[idx] = np.mean(inv_eps)
            else:
                x = pixel_coords[:, :, 0]
                y = pixel_coords[:, :, 1]
                angles = -2.0 * math.pi * (h * x + k * y)
                coeffs_solver[idx] = np.sum(inv_eps * np.exp(1j * angles)) / (size * size)
                
        # Lookup for differences
        G_lookup_solver = {}
        for idx, (h, k) in enumerate(G_vectors_solver):
            G_lookup_solver[(h, k)] = idx
            
        A = np.zeros((N_G_solver, N_G_solver), dtype=complex)
        
        for i_idx, (hi, ki) in enumerate(G_vectors_solver):
            # k_total = k + G_i
            kx_tot = kx + hi
            ky_tot = ky + ki
            k_sq = kx_tot * kx_tot + ky_tot * ky_tot
            
            for j_idx, (hj, kj) in enumerate(G_vectors_solver):
                diff_h = hi - hj
                diff_k = ki - kj
                # Check if diff is in our solver basis
                if (diff_h, diff_k) in G_lookup_solver:
                    diff_idx = G_lookup_solver[(diff_h, diff_k)]
                    # 1/eps(diff)
                    A[i_idx, j_idx] = k_sq * coeffs_solver[diff_idx]
                else:
                    # Should not happen if basis is large enough
                    A[i_idx, j_idx] = 0.0
                    
        # Solve eigenvalues
        # A is Hermitian? 
        # A_{ij} = k_sq_i * c_{i-j}. 
        # c_{-d} = c_d^* (for real inv_eps).
        # A_{ji} = k_sq_j * c_{j-i} = k_sq_j * c_{-(i-j)}^* = k_sq_j * c_{i-j}^*
        # A_{ij}^* = k_sq_i * c_{i-j}^*
        # They are not necessarily equal unless k_sq_i = k_sq_j.
        # Wait, the standard formulation for TM is:
        # sum_j (k+G_i)^2 / eps(G_i - G_j) H_j = omega^2 H_i
        # This matrix is not Hermitian in general?
        # Actually, the operator is (1/eps) * (k+G)^2.
        # In the basis of plane waves, the matrix elements are:
        # M_{ij} = (k+G_i)^2 * (1/eps)_{G_i - G_j}
        # Since (1/eps) is real, its Fourier coeffs are Hermitian: c_{-d} = c_d^*.
        # M_{ji} = (k+G_j)^2 * c_{j-i} = (k+G_j)^2 * c_{i-j}^*
        # M_{ij}^* = (k+G_i)^2 * c_{i-j}^*
        # M is not Hermitian. But it is similar to a Hermitian matrix?
        # Actually, for TM, the eigenvalues are real.
        # We can use np.linalg.eigvalsh if we make it Hermitian?
        # Or just use np.linalg.eigvals and take real parts.
        
        eigvals = np.linalg.eigvals(A)
        eigvals_real = np.real(eigvals)
        eigvals_real = np.sort(eigvals_real)
        
        # We need bands 4 and 5 (indices 3 and 4 in 0-based)
        # But we need the min of band 5 and max of band 4 over the BZ.
        # This function returns eigenvalues at one k.
        
        return eigvals_real[:5] # Return first 5

    # 3. Optimization Loop
    best_gap = -1.0
    best_cell = cell.copy()
    
    # Define k-points to scan
    # Irreducible BZ for square lattice: Triangle Gamma-M-X-Gamma
    # Gamma: (0,0)
    # M: (0.5, 0.5)
    # X: (0.5, 0)
    # We scan a few points in between.
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.5, 0.25),
        (0.1, 0.1),
        (0.4, 0.4)
    ]

    def evaluate_gap(pixel_grid):
        """
        Evaluates the gap-midgap ratio for the given pixel grid.
        """
        min_omega_5 = float('inf')
        max_omega_4 = -float('inf')
        
        for kx, ky in k_points:
            # Note: k is in units of 2pi/a.
            # Our G vectors are in units of 2pi/a (integers).
            # So kx, ky are consistent.
            
            eigs = compute_bands_at_k(kx, ky, pixel_grid)
            if len(eigs) < 5:
                return -1.0
            
            omega_4 = eigs[3]
            omega_5 = eigs[4]
            
            if omega_4 > max_omega_4:
                max_omega_4 = omega_4
            if omega_5 < min_omega_5:
                min_omega_5 = omega_5
                
        if min_omega_5 == float('inf') or max_omega_4 == -float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = min_omega_5 + max_omega_4
        if midgap == 0:
            return 0.0
        ratio = 2.0 * gap / midgap
        return ratio

    # Initial evaluation
    current_gap = evaluate_gap(cell)
    if current_gap > best_gap:
        best_gap = current_gap
        best_cell = cell.copy()
        
    # Local Search
    # We modify pixels in the top-left quadrant (0..15, 0..15)
    # and enforce symmetry.
    
    iterations = 0
    while time.time() < deadline:
        improved = False
        # Randomize the order of pixels to avoid getting stuck in local minima
        pixels_to_try = np.random.permutation((size // 2) * (size // 2))
        
        for p_idx in pixels_to_try:
            if time.time() > deadline - 5.0:
                break
                
            i = p_idx // (size // 2)
            j = p_idx % (size // 2)
            
            # Flip pixel
            old_val = cell[i, j]
            cell[i, j] = 1 - old_val
            
            # Enforce symmetry
            sym_cell = enforce_symmetry(cell)
            
            # Evaluate
            new_gap = evaluate_gap(sym_cell)
            
            if new_gap > current_gap:
                current_gap = new_gap
                best_gap = new_gap
                best_cell = sym_cell.copy()
                improved = True
            else:
                # Revert
                cell[i, j] = old_val
                
        if not improved:
            # Try a few random restarts or larger moves?
            # For now, just break if no improvement in a full pass
            # Or we can try flipping multiple pixels?
            # Let's try a few random restarts from the best cell
            for _ in range(3):
                if time.time() > deadline - 5.0:
                    break
                # Perturb best_cell
                perturbed = best_cell.copy()
                for _ in range(5):
                    ri = np.random.randint(0, size // 2)
                    rj = np.random.randint(0, size // 2)
                    perturbed[ri, rj] = 1 - perturbed[ri, rj]
                perturbed = enforce_symmetry(perturbed)
                p_gap = evaluate_gap(perturbed)
                if p_gap > current_gap:
                    current_gap = p_gap
                    best_gap = p_gap
                    best_cell = perturbed.copy()
                    improved = True
                    break
            if not improved:
                break

    # Return best_cell as nested list
    return best_cell.astype(int).tolist()

# EVOLVE-BLOCK-END
