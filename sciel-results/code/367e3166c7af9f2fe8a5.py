import numpy as np
import time

def _apply_symmetry(cell_quarter, size=32):
    """Apply 4-fold symmetry (D4) to a quarter-cell to produce full cell."""
    # cell_quarter is the top-left quarter (rows 0:16, cols 0:16)
    # Full cell has symmetry: cell[r,c] == cell[r,size-1-c] == cell[size-1-r,c] == cell[size-1-r,size-1-c]
    # Also transpose symmetry: cell[r,c] == cell[c,r]
    # So we only need the top-left triangle of the top-left quarter
    # Actually with full D4 symmetry about center:
    # cell[r,c] = cell[r, size-1-c] (horizontal mirror)
    # cell[r,c] = cell[size-1-r, c] (vertical mirror)  
    # cell[r,c] = cell[size-1-r, size-1-c] (180 rot)
    # cell[r,c] = cell[c, r] (transpose)
    # These imply: cell[r,c] = cell[c, size-1-r] = cell[size-1-r, size-1-c] = cell[size-1-c, r]
    
    cell = np.zeros((size, size), dtype=int)
    half = size // 2  # 16
    
    for r in range(half):
        for c in range(half):
            val = cell_quarter[r, c]
            # Apply all 8 symmetry operations
            cell[r, c] = val
            cell[r, size-1-c] = val
            cell[size-1-r, c] = val
            cell[size-1-r, size-1-c] = val
            cell[c, r] = val
            cell[c, size-1-r] = val
            cell[size-1-r, c] = val  # already set
            cell[size-1-r, size-1-c] = val  # already set
    
    return cell

def _compute_epsilon_fourier(cell, n_pw=17):
    """Compute Fourier coefficients of epsilon for plane-wave expansion.
    
    For TM polarization, we need 1/epsilon in the eigenvalue problem.
    We use a plane-wave expansion with n_pw x n_pw G-vectors.
    """
    size = cell.shape[0]
    eps_high = 12.0
    eps_low = 1.0
    
    # epsilon(r) = eps_low + (eps_high - eps_low) * cell(r)
    # For TM: we need 1/eps(r)
    # 1/eps(r) = sum_G G_G * exp(iG.r)
    
    # G-vectors: for a square lattice with period 1, G = (m, n) * 2*pi
    # We use G-vectors from -(n_pw-1)//2 to (n_pw-1)//2
    n_half = n_pw // 2
    G_indices = np.arange(-n_half, n_half + 1)
    
    # Create G-vector array
    Gx = G_indices * 2.0 * np.pi
    Gy = G_indices * 2.0 * np.pi
    GX, GY = np.meshgrid(Gx, Gy, indexing='ij')
    G_flat = np.stack([GX.flatten(), GY.flatten()], axis=1)
    G_idx_flat = np.stack([G_indices, G_indices], axis=1)
    
    # Compute Fourier coefficients of 1/eps
    # f(r) = 1/eps(r)
    # f_G = (1/N^2) * sum_{i,j} f(r_ij) * exp(-i G . r_ij)
    
    # Pixel centers
    x = (np.arange(size) + 0.5) / size
    y = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 1/epsilon at each pixel
    inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
    
    # Compute Fourier coefficients
    # f_G = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i*(Gx_i*x_i + Gy_j*y_j))
    # This can be done as a 2D FFT-like operation
    
    # For each G = (m*2pi, n*2pi):
    # f_G = (1/N^2) * sum_i sum_j inv_eps[i,j] * exp(-i*m*2pi*x_i) * exp(-i*n*2pi*y_j)
    # = (1/N^2) * [sum_i (sum_j inv_eps[i,j] * exp(-i*n*2pi*y_j)) * exp(-i*m*2pi*x_i)]
    
    # Use 2D DFT
    # f_G for G=(m,n) = (1/N^2) * DFT2D(inv_eps)[-m mod N, -n mod N]
    
    # Compute 2D FFT
    fft_2d = np.fft.fft2(inv_eps)
    
    # Map G indices to FFT indices
    # G = m*2pi corresponds to FFT frequency m
    # But we want negative frequencies too
    # FFT index for frequency m is m mod N
    # For m in [-n_half, n_half], the FFT index is m % N
    
    f_G = np.zeros((n_pw, n_pw), dtype=complex)
    for i, m in enumerate(G_indices):
        for j, n in enumerate(G_indices):
            fft_idx_i = m % size
            fft_idx_j = n % size
            f_G[i, j] = fft_2d[fft_idx_i, fft_idx_j] / (size * size)
    
    # Now build the matrix for TM bands
    # For TM: eigenvalue equation is:
    # sum_G' (G - G')^2 / (4*pi^2) * f_{G-G'} * c_{G'} = omega^2 * c_G
    # where f_{G-G'} is the Fourier coefficient of 1/eps
    
    # Build the matrix
    n = n_pw
    mat = np.zeros((n*n, n*n), dtype=complex)
    
    # Precompute G differences
    G_list = []
    for i in range(n):
        for j in range(n):
            G_list.append((G_indices[i] * 2.0 * np.pi, G_indices[j] * 2.0 * np.pi))
    
    for a in range(n*n):
        Ga = G_list[a]
        for b in range(n*n):
            Gb = G_list[b]
            dGx = Ga[0] - Gb[0]
            dGy = Ga[1] - Gb[1]
            # Find index of (dGx/(2pi), dGy/(2pi)) in G_indices
            mg = int(round(dGx / (2.0 * np.pi)))
            ng = int(round(dGy / (2.0 * np.pi)))
            if -n_half <= mg <= n_half and -n_half <= ng <= n_half:
                # Find the index in f_G
                idx_m = G_indices.tolist().index(mg)
                idx_n = G_indices.tolist().index(ng)
                f_val = f_G[idx_m, idx_n]
            else:
                f_val = 0.0
            
            mat[a, b] = (dGx**2 + dGy**2) * f_val
    
    return mat

def _compute_gap(cell, n_pw=13, n_bands=6, n_kpts=5):
    """Compute the gap between bands 4 and 5 for TM polarization.
    
    Returns the gap-midgap ratio: 2*(min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
    """
    try:
        mat = _compute_epsilon_fourier(cell, n_pw=n_pw)
    except Exception:
        return -1.0
    
    # Sample k-points in the irreducible Brillouin zone
    # For square lattice, IBZ is: 0 <= kx <= ky, kx + ky <= 1 (in units of 2pi)
    # Key points: Gamma(0,0), X(1,0), M(1,1), and along X-M line
    
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # along Gamma-X
        (1.0, 0.0),      # X
        (0.5, 0.5),      # along X-M (midpoint)
        (1.0, 1.0),      # M
    ]
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_points:
        # k in units of 2pi/a, so actual k = (kx*2pi, ky*2pi)
        k_actual = np.array([kx * 2.0 * np.pi, ky * 2.0 * np.pi])
        
        # Build the eigenvalue matrix at this k-point
        # The matrix elements are:
        # H_{G,G'} = (k+G)^2 / (4*pi^2) * delta_{G,G'} + sum_{G''} (G-G'')^2/(4*pi^2) * f_{G-G''} * delta_{G',G''}
        # Wait, let me reconsider the TM eigenvalue equation.
        
        # For TM polarization in 2D photonic crystal:
        # The eigenvalue equation is:
        # (1/eps(r)) * (-nabla^2) * E_z(r) = omega^2 * E_z(r)
        # In Fourier space:
        # sum_G' (k+G)^2 * f_{G-G'} * c_{G'} = omega^2 * c_G
        # where f_G are Fourier coefficients of 1/eps
        
        n = n_pw
        n_half = n // 2
        G_indices = np.arange(-n_half, n_half + 1)
        
        H = np.zeros((n*n, n*n), dtype=complex)
        
        G_list = []
        for i in range(n):
            for j in range(n):
                G_list.append((G_indices[i] * 2.0 * np.pi, G_indices[j] * 2.0 * np.pi))
        
        # Precompute f_G
        size = cell.shape[0]
        eps_high = 12.0
        eps_low = 1.0
        x = (np.arange(size) + 0.5) / size
        y = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(x, y, indexing='ij')
        inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
        fft_2d = np.fft.fft2(inv_eps)
        
        f_G = np.zeros((n, n), dtype=complex)
        for i, m in enumerate(G_indices):
            for j, ng in enumerate(G_indices):
                fft_idx_i = m % size
                fft_idx_j = ng % size
                f_G[i, j] = fft_2d[fft_idx_i, fft_idx_j] / (size * size)
        
        for a in range(n*n):
            Ga = G_list[a]
            k_plus_Ga = k_actual + np.array(Ga)
            kp2 = k_plus_Ga[0]**2 + k_plus_Ga[1]**2
            for b in range(n*n):
                Gb = G_list[b]
                dGx = Ga[0] - Gb[0]
                dGy = Ga[1] - Gb[1]
                mg = int(round(dGx / (2.0 * np.pi)))
                ng = int(round(dGy / (2.0 * np.pi)))
                if -n_half <= mg <= n_half and -n_half <= ng <= n_half:
                    idx_m = G_indices.tolist().index(mg)
                    idx_n = G_indices.tolist().index(ng)
                    f_val = f_G[idx_m, idx_n]
                else:
                    f_val = 0.0
                H[a, b] = kp2 * f_val
        
        # Solve eigenvalue problem
        try:
            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues = np.sort(np.real(eigenvalues))
        except Exception:
            continue
        
        if len(eigenvalues) >= n_bands:
            omega_4 = eigenvalues[3]  # 4th band (0-indexed: 3)
            omega_5 = eigenvalues[4]  # 5th band (0-indexed: 4)
            
            if omega_4 > max_omega_4:
                max_omega_4 = omega_4
            if omega_5 < min_omega_5:
                min_omega_5 = omega_5
    
    if max_omega_4 == -np.inf or min_omega_5 == np.inf:
        return -1.0
    
    if min_omega_5 <= max_omega_4:
        return -1.0  # No gap
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap <= 0:
        return -1.0
    
    return 2.0 * gap / midgap

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses local search to optimize the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem["pixels"]
    half = size // 2  # 16
    
    # Start with the 4-rod design
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    cell = np.zeros((size, size), dtype=int)
    dx = 0.25
    dy = 0.25
    R_rod = 0.12
    
    dist1 = np.sqrt((X - dx)**2 + (Y - dy)**2)
    cell[dist1 <= R_rod] = 1
    dist2 = np.sqrt((X + dx)**2 + (Y - dy)**2)
    cell[dist2 <= R_rod] = 1
    dist3 = np.sqrt((X - dx)**2 + (Y + dy)**2)
    cell[dist3 <= R_rod] = 1
    dist4 = np.sqrt((X + dx)**2 + (Y + dy)**2)
    cell[dist4 <= R_rod] = 1
    
    # Extract the independent quarter (top-left 16x16)
    # Due to symmetry, we only need to store and optimize this quarter
    quarter = cell[:half, :half].copy()
    
    best_score = _compute_gap(cell, n_pw=11, n_bands=6, n_kpts=5)
    best_quarter = quarter.copy()
    
    # Local search: try flipping pixels in the quarter
    # Each flip in the quarter affects 8 pixels in the full cell (due to symmetry)
    
    # Get list of independent positions
    # For D4 symmetry with transpose, the independent region is the upper triangle
    # of the top-left quarter (including diagonal)
    independent_positions = []
    for r in range(half):
        for c in range(half):
            if c <= r:  # upper triangle including diagonal
                independent_positions.append((r, c))
    
    # Also need to consider that some positions may be equivalent under symmetry
    # but for now, let's just iterate over all positions in the quarter
    
    # Actually, let me reconsider. The symmetry constraints are:
    # cell[r,c] = cell[r, size-1-c] = cell[size-1-r, c] = cell[size-1-r, size-1-c]
    # cell[r,c] = cell[c, r] = cell[c, size-1-r] = cell[size-1-r, size-1-c]
    # So the independent variables are those in the top-left triangle where r >= c
    # and r < half, c < half (to avoid double-counting with the other quarters)
    
    # Let me just use all positions in the top-left quarter and enforce symmetry after
    all_positions = [(r, c) for r in range(half) for c in range(half)]
    
    # Greedy local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r, c in all_positions:
            if time.time() > deadline:
                break
            
            # Try flipping pixel (r, c) in the quarter
            new_quarter = quarter.copy()
            new_quarter[r, c] = 1 - new_quarter[r, c]
            
            # Apply symmetry to get full cell
            new_cell = _apply_symmetry(new_quarter, size)
            
            # Evaluate
            score = _compute_gap(new_cell, n_pw=11, n_bands=6, n_kpts=5)
            
            if score > best_score:
                best_score = score
                best_quarter = new_quarter.copy()
                quarter = new_quarter
                improved = True
    
    # Final evaluation with more accurate settings
    final_cell = _apply_symmetry(best_quarter, size)
    final_score = _compute_gap(final_cell, n_pw=13, n_bands=6, n_kpts=5)
    
    return final_cell.tolist()
