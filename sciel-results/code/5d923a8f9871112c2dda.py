import numpy as np
import time

def _apply_symmetry(cell_quarter, size=32):
    """Apply 4-fold symmetry (D4) to a quarter-cell to produce full cell.
    The input cell_quarter is the top-left 16x16 block.
    It must be symmetric (Q == Q.T). We enforce this first.
    """
    # Enforce symmetry of the quarter
    # Q[r,c] = Q[c,r]
    # We take the upper triangle and reflect it
    q = np.array(cell_quarter)
    # Average to handle potential inconsistencies or just take upper
    # Since it's binary, we can just use upper triangle
    q_sym = np.triu(q) + np.triu(q, 1).T
    
    cell = np.zeros((size, size), dtype=int)
    half = size // 2  # 16
    
    # Fill top-left
    cell[:half, :half] = q_sym
    
    # Fill top-right (horizontal flip of top-left)
    cell[:half, half:] = q_sym[:, ::-1]
    
    # Fill bottom-left (vertical flip of top-left)
    cell[half:, :half] = q_sym[::-1, :]
    
    # Fill bottom-right (180 rot of top-left)
    cell[half:, half:] = q_sym[::-1, ::-1]
    
    return cell

def _compute_gap_fast(cell, n_pw=11, n_kpts=3):
    """Compute gap-midgap ratio using a fast approximation."""
    size = cell.shape[0]
    eps_high = 12.0
    eps_low = 1.0
    
    # Compute Fourier coefficients of 1/epsilon
    # f_G = (1/N^2) sum inv_eps[i,j] exp(-i G . r_ij)
    # Use FFT
    inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
    fft_2d = np.fft.fft2(inv_eps)
    
    n_half = n_pw // 2
    G_indices = np.arange(-n_half, n_half + 1)
    
    # Precompute f_G for the required G-vectors
    f_G = np.zeros((n_pw, n_pw), dtype=complex)
    for i, m in enumerate(G_indices):
        for j, n in enumerate(G_indices):
            fft_idx_i = m % size
            fft_idx_j = n % size
            f_G[i, j] = fft_2d[fft_idx_i, fft_idx_j] / (size * size)
    
    # Precompute G-vector magnitudes and coordinates
    G_list = []
    for i in range(n_pw):
        for j in range(n_pw):
            G_list.append((G_indices[i] * 2.0 * np.pi, G_indices[j] * 2.0 * np.pi))
            
    # Precompute the matrix F where F[i,j] = f_{G_i - G_j}
    # Note: G_i - G_j corresponds to index (i-j) in G_indices?
    # G_i = (m_i, n_i) * 2pi
    # G_j = (m_j, n_j) * 2pi
    # G_i - G_j = (m_i - m_j, n_i - n_j) * 2pi
    # We need f_{(m_i - m_j, n_i - n_j)}
    
    F = np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=complex)
    
    for i in range(n_pw*n_pw):
        mi = G_indices[i % n_pw]
        ni = G_indices[i // n_pw]
        for j in range(n_pw*n_pw):
            mj = G_indices[j % n_pw]
            nj = G_indices[j // n_pw]
            dm = mi - mj
            dn = ni - nj
            # Check if (dm, dn) is in our G set
            if -n_half <= dm <= n_half and -n_half <= dn <= n_half:
                idx_m = G_indices.tolist().index(dm)
                idx_n = G_indices.tolist().index(dn)
                F[i, j] = f_G[idx_m, idx_n]
            else:
                F[i, j] = 0.0
                
    # Define k-points in the IBZ
    # Gamma (0,0), X (1,0), M (1,1) in units of 2pi/a
    k_points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_points:
        k_actual = np.array([kx * 2.0 * np.pi, ky * 2.0 * np.pi])
        
        # Compute v_i = |k + G_i|
        v = np.zeros(n_pw*n_pw)
        for i in range(n_pw*n_pw):
            Gx = G_list[i][0]
            Gy = G_list[i][1]
            v[i] = np.sqrt((k_actual[0] + Gx)**2 + (k_actual[1] + Gy)**2)
        
        # H = diag(v) F diag(v)
        # H[i,j] = v[i] * F[i,j] * v[j]
        H = F * v[:, None] * v[None, :]
        
        # Solve eigenvalue problem
        try:
            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues = np.sort(np.real(eigenvalues))
        except Exception:
            continue
        
        if len(eigenvalues) >= 5:
            omega_4 = eigenvalues[3]
            omega_5 = eigenvalues[4]
            
            if omega_4 > max_omega_4:
                max_omega_4 = omega_4
            if omega_5 < min_omega_5:
                min_omega_5 = omega_5
                
    if max_omega_4 == -np.inf or min_omega_5 == np.inf:
        return -1.0
        
    if min_omega_5 <= max_omega_4:
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap <= 0:
        return -1.0
        
    return 2.0 * gap / midgap

def _compute_gap_accurate(cell, n_pw=17, n_kpts=5):
    """Compute gap-midgap ratio using a more accurate model."""
    # Reuse the logic but with higher n_pw and more k-points
    # For simplicity, we can just call _compute_gap_fast with higher params
    # but _compute_gap_fast is hardcoded to 3 k-points.
    # Let's create a generic version or just call the fast one with adjusted params if we refactor.
    # Given time constraints, we'll stick to the fast one for search and a slightly better one for final.
    
    # Actually, let's just use n_pw=15 and 5 k-points for final.
    # We need a version of _compute_gap that accepts n_kpts and k_points list.
    
    size = cell.shape[0]
    eps_high = 12.0
    eps_low = 1.0
    
    inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
    fft_2d = np.fft.fft2(inv_eps)
    
    n_half = n_pw // 2
    G_indices = np.arange(-n_half, n_half + 1)
    
    f_G = np.zeros((n_pw, n_pw), dtype=complex)
    for i, m in enumerate(G_indices):
        for j, n in enumerate(G_indices):
            fft_idx_i = m % size
            fft_idx_j = n % size
            f_G[i, j] = fft_2d[fft_idx_i, fft_idx_j] / (size * size)
            
    G_list = []
    for i in range(n_pw):
        for j in range(n_pw):
            G_list.append((G_indices[i] * 2.0 * np.pi, G_indices[j] * 2.0 * np.pi))
            
    F = np.zeros((n_pw*n_pw, n_pw*n_pw), dtype=complex)
    for i in range(n_pw*n_pw):
        mi = G_indices[i % n_pw]
        ni = G_indices[i // n_pw]
        for j in range(n_pw*n_pw):
            mj = G_indices[j % n_pw]
            nj = G_indices[j // n_pw]
            dm = mi - mj
            dn = ni - nj
            if -n_half <= dm <= n_half and -n_half <= dn <= n_half:
                idx_m = G_indices.tolist().index(dm)
                idx_n = G_indices.tolist().index(dn)
                F[i, j] = f_G[idx_m, idx_n]
            else:
                F[i, j] = 0.0
                
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ]
    
    min_omega_5 = np.inf
    max_omega_4 = -np.inf
    
    for kx, ky in k_points:
        k_actual = np.array([kx * 2.0 * np.pi, ky * 2.0 * np.pi])
        v = np.zeros(n_pw*n_pw)
        for i in range(n_pw*n_pw):
            Gx = G_list[i][0]
            Gy = G_list[i][1]
            v[i] = np.sqrt((k_actual[0] + Gx)**2 + (k_actual[1] + Gy)**2)
        
        H = F * v[:, None] * v[None, :]
        
        try:
            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues = np.sort(np.real(eigenvalues))
        except Exception:
            continue
        
        if len(eigenvalues) >= 5:
            omega_4 = eigenvalues[3]
            omega_5 = eigenvalues[4]
            
            if omega_4 > max_omega_4:
                max_omega_4 = omega_4
            if omega_5 < min_omega_5:
                min_omega_5 = omega_5
                
    if max_omega_4 == -np.inf or min_omega_5 == np.inf:
        return -1.0
        
    if min_omega_5 <= max_omega_4:
        return -1.0
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap <= 0:
        return -1.0
        
    return 2.0 * gap / midgap

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0
    
    size = problem["pixels"]
    half = size // 2
    
    # Initial design: 4 rods
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
    
    quarter = cell[:half, :half].copy()
    
    best_score = _compute_gap_fast(cell, n_pw=11, n_kpts=3)
    best_quarter = quarter.copy()
    
    # Local search
    # Independent positions: upper triangle of 16x16
    independent_positions = []
    for r in range(half):
        for c in range(half):
            if c <= r:
                independent_positions.append((r, c))
                
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r, c in independent_positions:
            if time.time() > deadline:
                break
            
            new_quarter = quarter.copy()
            new_quarter[r, c] = 1 - new_quarter[r, c]
            
            new_cell = _apply_symmetry(new_quarter, size)
            
            score = _compute_gap_fast(new_cell, n_pw=11, n_kpts=3)
            
            if score > best_score:
                best_score = score
                best_quarter = new_quarter.copy()
                quarter = new_quarter
                improved = True
                
    final_cell = _apply_symmetry(best_quarter, size)
    
    # Final evaluation
    final_score = _compute_gap_accurate(final_cell, n_pw=17, n_kpts=5)
    
    return final_cell.tolist()
