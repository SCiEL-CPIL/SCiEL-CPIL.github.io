<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
>>>>>>> REPLACE
<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def _compute_gap(cell, n_pw=17, n_k=8):
    """
    Fast approximate TM band gap calculation using plane-wave expansion.
    Returns the gap-midgap ratio between bands 4 and 5.
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    
    # Fourier coefficients of epsilon
    # Use a subset of plane waves for speed
    max_g = n_pw  # max G vector index
    
    # Generate G vectors
    G_vals = []
    for gx in range(-max_g, max_g + 1):
        for gy in range(-max_g, max_g + 1):
            G_vals.append((gx, gy))
    G_vals = np.array(G_vals, dtype=float)
    n_g = len(G_vals)
    
    # Compute Fourier coefficients of 1/eps for TM mode
    # eps_inv = 1/eps
    eps_inv = 1.0 / (1.0 + 11.0 * eps)  # 1/eps where eps is 1 or 12
    
    # FFT to get Fourier coefficients
    # Pad to size x size
    eps_inv_padded = np.zeros((size, size))
    eps_inv_padded[:size, :size] = eps_inv
    
    # Compute Fourier coefficients
    # The Fourier series: eps_inv(k) = sum_G eps_inv_G exp(iG.r)
    # eps_inv_G = (1/N^2) sum_{n,m} eps_inv(n,m) exp(-iG.r_n)
    # For a pixel grid, this is essentially the DFT
    
    # Use FFT for efficiency
    fft_2d = np.fft.fft2(eps_inv_padded)
    
    # Map G vectors to FFT indices
    # G = (gx, gy) corresponds to FFT index (gx mod size, gy mod size)
    def g_to_fft_idx(gx, gy, size):
        return (gx % size, gy % size)
    
    # Build the eps_inv matrix for the plane-wave basis
    eps_inv_matrix = np.zeros((n_g, n_g), dtype=complex)
    for i, (gx, gy) in enumerate(G_vals):
        for j, (gx2, gy2) in enumerate(G_vals):
            diff_gx = gx - gx2
            diff_gy = gy - gy2
            # Only non-zero if diff is a multiple of size (aliasing)
            # For our G range, this is only when diff_gx == 0 and diff_gy == 0
            # Actually, we need to be more careful with the pixel discretization
            # For simplicity, use the FFT coefficients directly
            pass
    
    # Alternative: direct computation for small G set
    # Let's use a simpler approach with direct summation for accuracy
    # But that's slow. Let's use the FFT approach properly.
    
    # The dielectric function in Fourier space:
    # For TM mode, the eigenvalue problem is:
    # sum_G' (G + G')^2 eps_inv(G - G') E(G') = omega^2 E(G)
    
    # We need eps_inv(G) for G in the set of differences of our G vectors
    # The differences range from -2*max_g to 2*max_g
    
    # Compute eps_inv Fourier coefficients up to the needed range
    max_diff_g = 2 * max_g
    fft_size = size  # Use the actual pixel size
    
    # Get Fourier coefficients from FFT
    # fft_2d[gx, gy] gives coefficient for G = (gx, gy) where gx, gy are in [-size/2, size/2)
    def get_eps_inv_coeff(gx, gy):
        """Get Fourier coefficient for G = (gx, gy)"""
        # Map to FFT index
        ix = gx % size
        iy = gy % size
        return fft_2d[ix, iy] / (size * size)
    
    # Build the matrix
    eps_inv_mat = np.zeros((n_g, n_g), dtype=complex)
    for i, (gx, gy) in enumerate(G_vals):
        for j, (gx2, gy2) in enumerate(G_vals):
            dgx = gx - gx2
            dgy = gy - gy2
            eps_inv_mat[i, j] = get_eps_inv_coeff(dgx, dgy)
    
    # Define K points in the irreducible Brillouin zone
    # Square lattice, irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= 1
    # Key points: Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Let's sample along the high-symmetry lines and some interior points
    k_points = [
        (0.0, 0.0),       # Gamma
        (0.25, 0.0),
        (0.5, 0.0),       # M
        (0.25, 0.25),
        (0.5, 0.25),
        (0.3536, 0.3536), # X direction (1/sqrt(2), 1/sqrt(2)) * 0.5
        (0.5, 0.5),       # X
    ]
    
    # For each K point, solve the eigenvalue problem
    min_omega_5 = np.inf
    max_omega_4 = 0.0
    
    for kx, ky in k_points:
        # Build the kinetic energy matrix
        # T(G, G') = (G + K)^2 * delta(G, G')
        # Actually, for plane waves: (G + K)^2
        eigvals = np.zeros(n_g)
        for i, (gx, gy) in enumerate(G_vals):
            kx_total = kx + gx
            ky_total = ky + gy
            eigvals[i] = kx_total * kx_total + ky_total * ky_total
        
        # The eigenvalue problem: sum_j (G_i + K)^2 eps_inv(G_i - G_j) E_j = omega^2 E_i
        # This is a generalized eigenvalue problem or we can convert
        # A E = omega^2 E where A_ij = (G_i + K)^2 eps_inv(G_i - G_j)
        # But this is not symmetric in general. We need to be careful.
        
        # For TM mode, the equation is:
        # sum_G' (G + G')^2 eps_inv(G - G') E(G') = omega^2 E(G)
        # where G here means G + K
        
        # Let me redefine: let K be the crystal momentum, and G be reciprocal lattice vectors
        # The plane waves are exp(i(K+G).r)
        # The equation becomes:
        # sum_{G'} (K + G + G')^2 eps_inv(G - G') E(G') = omega^2 E(G)
        
        # This is a standard form. Let's build the matrix properly.
        A = np.zeros((n_g, n_g), dtype=complex)
        for i, (gx, gy) in enumerate(G_vals):
            for j, (gx2, gy2) in enumerate(G_vals):
                # (K + G_i + G_j)^2
                kx_total = kx + gx + gx2
                ky_total = ky + gy + gy2
                k_sq = kx_total * kx_total + ky_total * ky_total
                A[i, j] = k_sq * eps_inv_mat[i, j]
        
        # Solve eigenvalue problem
        # Use numpy's eig for non-symmetric matrices
        try:
            vals, vecs = np.linalg.eig(A)
            # Take real parts and sort
            vals_real = np.real(vals)
            # Filter out negative or very small values
            vals_real = vals_real[vals_real > 1e-6]
            vals_real = np.sort(vals_real)
            
            if len(vals_real) >= 5:
                omega_4 = np.sqrt(vals_real[3])  # 4th band (0-indexed: 3)
                omega_5 = np.sqrt(vals_real[4])  # 5th band (0-indexed: 4)
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        except:
            continue
    
    if min_omega_5 == np.inf or max_omega_4 == 0:
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return -1.0
    
    return gap / midgap
=======
# EVOLVE-BLOCK-START
def _compute_gap(cell, n_pw=17, n_k=8):
    """
    Fast approximate TM band gap calculation using plane-wave expansion.
    Returns the gap-midgap ratio between bands 4 and 5.
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    
    # Fourier coefficients of epsilon
    # Use a subset of plane waves for speed
    max_g = n_pw  # max G vector index
    
    # Generate G vectors
    G_vals = []
    for gx in range(-max_g, max_g + 1):
        for gy in range(-max_g, max_g + 1):
            G_vals.append((gx, gy))
    G_vals = np.array(G_vals, dtype=float)
    n_g = len(G_vals)
    
    # Compute Fourier coefficients of 1/eps for TM mode
    # eps_inv = 1/eps
    eps_inv = 1.0 / (1.0 + 11.0 * eps)  # 1/eps where eps is 1 or 12
    
    # FFT to get Fourier coefficients
    # Pad to size x size
    eps_inv_padded = np.zeros((size, size))
    eps_inv_padded[:size, :size] = eps_inv
    
    # Compute Fourier coefficients
    # The Fourier series: eps_inv(k) = sum_G eps_inv_G exp(iG.r)
    # eps_inv_G = (1/N^2) sum_{n,m} eps_inv(n,m) exp(-iG.r_n)
    # For a pixel grid, this is essentially the DFT
    
    # Use FFT for efficiency
    fft_2d = np.fft.fft2(eps_inv_padded)
    
    # Map G vectors to FFT indices
    # G = (gx, gy) corresponds to FFT index (gx mod size, gy mod size)
    def g_to_fft_idx(gx, gy, size):
        return (gx % size, gy % size)
    
    # Build the eps_inv matrix for the plane-wave basis
    eps_inv_matrix = np.zeros((n_g, n_g), dtype=complex)
    for i, (gx, gy) in enumerate(G_vals):
        for j, (gx2, gy2) in enumerate(G_vals):
            diff_gx = gx - gx2
            diff_gy = gy - gy2
            # Only non-zero if diff is a multiple of size (aliasing)
            # For our G range, this is only when diff_gx == 0 and diff_gy == 0
            # Actually, we need to be more careful with the pixel discretization
            # For simplicity, use the FFT coefficients directly
            pass
    
    # Alternative: direct computation for small G set
    # Let's use a simpler approach with direct summation for accuracy
    # But that's slow. Let's use the FFT approach properly.
    
    # The dielectric function in Fourier space:
    # For TM mode, the eigenvalue problem is:
    # sum_G' (G + G')^2 eps_inv(G - G') E(G') = omega^2 E(G)
    
    # We need eps_inv(G) for G in the set of differences of our G vectors
    # The differences range from -2*max_g to 2*max_g
    
    # Compute eps_inv Fourier coefficients up to the needed range
    max_diff_g = 2 * max_g
    fft_size = size  # Use the actual pixel size
    
    # Get Fourier coefficients from FFT
    # fft_2d[gx, gy] gives coefficient for G = (gx, gy) where gx, gy are in [-size/2, size/2)
    def get_eps_inv_coeff(gx, gy):
        """Get Fourier coefficient for G = (gx, gy)"""
        # Map to FFT index
        ix = gx % size
        iy = gy % size
        return fft_2d[ix, iy] / (size * size)
    
    # Build the matrix
    eps_inv_mat = np.zeros((n_g, n_g), dtype=complex)
    for i, (gx, gy) in enumerate(G_vals):
        for j, (gx2, gy2) in enumerate(G_vals):
            dgx = gx - gx2
            dgy = gy - gy2
            eps_inv_mat[i, j] = get_eps_inv_coeff(dgx, dgy)
    
    # Define K points in the irreducible Brillouin zone
    # Square lattice, irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= 1
    # Key points: Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Let's sample along the high-symmetry lines and some interior points
    k_points = [
        (0.0, 0.0),       # Gamma
        (0.25, 0.0),
        (0.5, 0.0),       # M
        (0.25, 0.25),
        (0.5, 0.25),
        (0.3536, 0.3536), # X direction (1/sqrt(2), 1/sqrt(2)) * 0.5
        (0.5, 0.5),       # X
    ]
    
    # For each K point, solve the eigenvalue problem
    min_omega_5 = np.inf
    max_omega_4 = 0.0
    
    for kx, ky in k_points:
        # Build the kinetic energy matrix
        # T(G, G') = (G + K)^2 * delta(G, G')
        # Actually, for plane waves: (G + K)^2
        eigvals = np.zeros(n_g)
        for i, (gx, gy) in enumerate(G_vals):
            kx_total = kx + gx
            ky_total = ky + gy
            eigvals[i] = kx_total * kx_total + ky_total * ky_total
        
        # The eigenvalue problem: sum_j (G_i + K)^2 eps_inv(G_i - G_j) E_j = omega^2 E_i
        # This is a generalized eigenvalue problem or we can convert
        # A E = omega^2 E where A_ij = (G_i + K)^2 eps_inv(G_i - G_j)
        # But this is not symmetric in general. We need to be careful.
        
        # For TM mode, the equation is:
        # sum_G' (G + G')^2 eps_inv(G - G') E(G') = omega^2 E(G)
        # where G here means G + K
        
        # Let me redefine: let K be the crystal momentum, and G be reciprocal lattice vectors
        # The plane waves are exp(i(K+G).r)
        # The equation becomes:
        # sum_{G'} (K + G + G')^2 eps_inv(G - G') E(G') = omega^2 E(G)
        
        # This is a standard form. Let's build the matrix properly.
        A = np.zeros((n_g, n_g), dtype=complex)
        for i, (gx, gy) in enumerate(G_vals):
            for j, (gx2, gy2) in enumerate(G_vals):
                # (K + G_i + G_j)^2
                kx_total = kx + gx + gx2
                ky_total = ky + gy + gy2
                k_sq = kx_total * kx_total + ky_total * ky_total
                A[i, j] = k_sq * eps_inv_mat[i, j]
        
        # Solve eigenvalue problem
        # Use numpy's eig for non-symmetric matrices
        try:
            vals, vecs = np.linalg.eig(A)
            # Take real parts and sort
            vals_real = np.real(vals)
            # Filter out negative or very small values
            vals_real = vals_real[vals_real > 1e-6]
            vals_real = np.sort(vals_real)
            
            if len(vals_real) >= 5:
                omega_4 = np.sqrt(vals_real[3])  # 4th band (0-indexed: 3)
                omega_5 = np.sqrt(vals_real[4])  # 5th band (0-indexed: 4)
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        except:
            continue
    
    if min_omega_5 == np.inf or max_omega_4 == 0:
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return -1.0
    
    return gap / midgap
>>>>>>> REPLACE
<<<<<<< SEARCH
def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 230s kill
    
    # Initial best solution
    best_cell = None
    best_score = -1.0
    
    # Generate initial candidates
    candidates = []
    
    # 1. Various circle radii
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append([r, 0.0, 0.0, 0])
    
    # 2. Squares
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append([r, 0.0, 0.0, 1])
    
    # 3. Diamonds
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append([r, 0.0, 0.0, 2])
    
    # 4. Offset shapes
    for r in [0.2, 0.3, 0.4]:
        for off in [0.05, 0.1]:
            candidates.append([r, off, 0.0, 0])
            candidates.append([r, 0.0, off, 0])
    
    # Evaluate candidates
    for params in candidates:
        if time.time() > deadline - 30:
            break
        
        cell = _cell_from_params(params, size)
        cell = _ensure_symmetry(cell, size)
        score = _compute_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # If we have time, do some local optimization
    if time.time() < deadline - 60 and best_cell is not None:
        # Try to improve the best candidate
        best_params = None
        # Find which candidate gave the best score
        for params in candidates:
            cell = _cell_from_params(params, size)
            cell = _ensure_symmetry(cell, size)
            if np.array_equal(cell, best_cell):
                best_params = params
                break
        
        if best_params is not None:
            # Local search around the best
            base = best_params
            for delta_r in [-0.02, -0.01, 0.01, 0.02]:
                for delta_x in [-0.02, 0.0, 0.02]:
                    for delta_y in [-0.02, 0.0, 0.02]:
                        if time.time() > deadline - 30:
                            break
                        
                        new_params = [
                            base[0] + delta_r,
                            base[1] + delta_x,
                            base[2] + delta_y,
                            base[3]
                        ]
                        
                        # Clamp radius
                        new_params[0] = max(0.05, min(0.5, new_params[0]))
                        
                        cell = _cell_from_params(new_params, size)
                        cell = _ensure_symmetry(cell, size)
                        score = _compute_gap(cell)
                        
                        if score > best_score:
                            best_score = score
                            best_cell = cell.copy()
    
    # Convert to list format
    if best_cell is None:
        # Fallback: default circle
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.2 ** 2:
                    best_cell[i, j] = 1
    
    return best_cell.tolist()
=======
def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 230s kill
    
    # Initial best solution
    best_cell = None
    best_score = -1.0
    
    # Generate initial candidates
    candidates = []
    
    # 1. Various circle radii
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append([r, 0.0, 0.0, 0])
    
    # 2. Squares
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append([r, 0.0, 0.0, 1])
    
    # 3. Diamonds
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append([r, 0.0, 0.0, 2])
    
    # 4. Offset shapes
    for r in [0.2, 0.3, 0.4]:
        for off in [0.05, 0.1]:
            candidates.append([r, off, 0.0, 0])
            candidates.append([r, 0.0, off, 0])
    
    # Evaluate candidates
    for params in candidates:
        if time.time() > deadline - 30:
            break
        
        cell = _cell_from_params(params, size)
        cell = _ensure_symmetry(cell, size)
        score = _compute_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # If we have time, do some local optimization
    if time.time() < deadline - 60 and best_cell is not None:
        # Try to improve the best candidate
        best_params = None
        # Find which candidate gave the best score
        for params in candidates:
            cell = _cell_from_params(params, size)
            cell = _ensure_symmetry(cell, size)
            if np.array_equal(cell, best_cell):
                best_params = params
                break
        
        if best_params is not None:
            # Local search around the best
            base = best_params
            for delta_r in [-0.02, -0.01, 0.01, 0.02]:
                for delta_x in [-0.02, 0.0, 0.02]:
                    for delta_y in [-0.02, 0.0, 0.02]:
                        if time.time() > deadline - 30:
                            break
                        
                        new_params = [
                            base[0] + delta_r,
                            base[1] + delta_x,
                            base[2] + delta_y,
                            base[3]
                        ]
                        
                        # Clamp radius
                        new_params[0] = max(0.05, min(0.5, new_params[0]))
                        
                        cell = _cell_from_params(new_params, size)
                        cell = _ensure_symmetry(cell, size)
                        score = _compute_gap(cell)
                        
                        if score > best_score:
                            best_score = score
                            best_cell = cell.copy()
    
    # Convert to list format
    if best_cell is None:
        # Fallback: default circle
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.2 ** 2:
                    best_cell[i, j] = 1
    
    return best_cell.tolist()
>>>>>>> REPLACE
