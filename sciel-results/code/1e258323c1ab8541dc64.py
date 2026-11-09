import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Precompute G vectors for 289 plane waves (17x17 grid, -8 to 8)
    N_G = 17
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # Precompute diff indices for Hamiltonian construction
    # diff_indices[m, n] is the index into the flat eps_inv_coeffs array for G_m - G_n
    # The coeffs array is indexed by (Gx + 8, Gy + 8) in a 17x17 grid, flattened.
    # eps_inv_coeffs will be a 1D array of length 289, ordered by Gx, Gy.
    # We need to map (Gx_m - Gx_n, Gy_m - Gy_n) to an index in [-8, 8] x [-8, 8].
    # If the difference is outside [-8, 8], the coefficient is 0 (truncation).
    
    Gx_flat = Gx.ravel()
    Gy_flat = Gy.ravel()
    
    # Create meshgrid of differences
    # dx[m, n] = Gx_m - Gx_n
    # dy[m, n] = Gy_m - Gy_n
    dx = Gx_flat[:, None] - Gx_flat[None, :]
    dy = Gy_flat[:, None] - Gy_flat[None, :]
    
    # Map to indices in the 17x17 grid centered at 0
    # Index in flat array: (dx + 8) + (dy + 8) * 17
    # But we need to handle out-of-bounds by setting to 0 or ignoring.
    # Since we multiply by 0 for out-of-bounds, we can just clip or mask.
    # Better: create a mask of valid indices.
    valid_mask = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
    
    # Initialize index matrix with -1 for invalid
    diff_indices = np.full((N_pw, N_pw), -1, dtype=int)
    diff_indices[valid_mask] = (dx[valid_mask] + N_G//2) + (dy[valid_mask] + N_G//2) * N_G
    
    # K points in irreducible BZ (approx 15-20 points for speed, or use all 30 if fast enough)
    # The evaluator uses 30 points. We should use a representative subset or all if fast.
    # With the fast solver, 30 points * 289x289 eigensolve is ~30 * 0.1ms = 3ms.
    # So we can afford 30 points.
    k_points = []
    # Boundary
    for i in range(6):
        k_points.append([i/5.0, 0.0])
        k_points.append([0.5, i/5.0])
        k_points.append([i/5.0, i/5.0])
    # Interior
    k_points.append([0.25, 0.1])
    k_points.append([0.4, 0.2])
    k_points.append([0.3, 0.15])
    k_points = np.array(k_points)
    
    def generate_cell(radius):
        """Generate a symmetric cell with a central dielectric rod of given radius."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < radius**2:
                    cell[i, j] = 1
        return cell

    def evaluate_cell(cell):
        # Compute inverse epsilon field
        eps = np.where(cell == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        
        # FFT to get Fourier coefficients
        coeffs = np.fft.fft2(eps_inv) / (size * size)
        # coeffs is 32x32. We need the 17x17 subset centered at 0.
        # The FFT output is ordered 0, 1, ..., 15, 16, ..., 31.
        # Frequency 0 is at index 0. Frequency -8 is at index 24 (32-8).
        # We need to extract the 17x17 block corresponding to Gx, Gy in [-8, 8].
        
        # Let's construct the 17x17 coeff array in the order matching G_vectors
        # G_vectors are ordered by Gx, then Gy? 
        # meshgrid(Gx, Gy, indexing='ij') -> Gx varies slow, Gy fast?
        # Gx.ravel() with 'ij': 
        # Row 0: Gx=-8, Gy=-8..8
        # Row 1: Gx=-7, Gy=-8..8
        # ...
        # So index = (Gx + 8) * 17 + (Gy + 8)
        
        coeff_subset = np.zeros((N_G, N_G), dtype=complex)
        # Map Gx, Gy to FFT indices
        # FFT index for Gx: if Gx >= 0, idx = Gx. If Gx < 0, idx = size + Gx.
        for gx_idx, gx in enumerate(g_range):
            fft_gx = gx % size
            for gy_idx, gy in enumerate(g_range):
                fft_gy = gy % size
                coeff_subset[gx_idx, gy_idx] = coeffs[fft_gx, fft_gy]
                
        # Flatten coeff_subset to match G_vectors order
        eps_inv_coeffs = coeff_subset.ravel()
        
        # Precompute coefficient matrix for this cell
        # C_matrix[m, n] = eps_inv_coeffs[diff_indices[m, n]]
        # diff_indices has -1 for invalid. We replace -1 with 0 and mask later, or just set coeffs to 0.
        # Let's create a safe index matrix
        safe_indices = diff_indices.copy()
        safe_indices[safe_indices == -1] = 0 # Dummy index
        C_matrix = eps_inv_coeffs[safe_indices]
        C_matrix[~valid_mask] = 0.0
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            # Compute (k + G_j)^2 for all j
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # H[m, n] = C_matrix[m, n] * k_plus_G_sq[n]
            H = C_matrix * k_plus_G_sq[None, :]
            
            # Solve eigenvalue problem
            # H is not Hermitian? 
            # eps_inv is real and symmetric in space? 
            # The matrix H is constructed from a convolution. 
            # In the plane wave basis, the operator is not Hermitian in general for TM?
            # Wait, the TM operator is (k+G)^2 / eps(r). 
            # The matrix elements are <k+G_m | (k+G_n)^2 / eps(r) | k+G_n> ?
            # No, the standard formulation is:
            # sum_n H_{mn} c_n = omega^2 c_m
            # H_{mn} = (k+G_n)^2 * eps_inv(G_m - G_n)
            # This matrix is not Hermitian.
            # However, the eigenvalues are real.
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
            # Get 4th and 5th bands (index 3 and 4)
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # Search over radius
    best_score = -1.0
    best_cell = None
    
    # Grid search
    radii = np.linspace(0.05, 0.45, 50)
    for r in radii:
        cell = generate_cell(r)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
    # Local refinement around best radius
    best_radius = None
    # Find best radius from grid
    for r in radii:
        if np.allclose(generate_cell(r), best_cell):
            best_radius = r
            break
            
    if best_radius is not None:
        # Refine
        fine_radii = np.linspace(best_radius - 0.02, best_radius + 0.02, 20)
        for r in fine_radii:
            if r < 0 or r > 0.5: continue
            cell = generate_cell(r)
            score = evaluate_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                
    # Return best_cell as nested list
    if best_cell is None:
        # Fallback to seed
        best_cell = generate_cell(0.2)
        
    return best_cell.tolist()
