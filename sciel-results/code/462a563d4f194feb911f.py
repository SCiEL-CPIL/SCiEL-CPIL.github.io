import numpy as np
import time

def _apply_symmetry(cell_quarter, size=32):
    """Apply 4-fold symmetry (D4) to a quarter-cell to produce full cell."""
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
            # Note: cell[size-1-r, c] and cell[size-1-r, size-1-c] are already set by previous lines
            # But we must ensure consistency if the quarter itself violates some sub-symmetries?
            # No, the quarter is the fundamental domain for D4 if we consider the full set of operations.
            # Actually, D4 has 8 elements. The fundamental domain is 1/8 of the cell.
            # The top-left quarter (16x16) is 1/4 of the cell.
            # Within the quarter, do we need more symmetry?
            # cell[r,c] == cell[c,r] is one of the symmetries.
            # So cell_quarter[r,c] must equal cell_quarter[c,r].
            # If we just copy quarter to full cell, we must ensure quarter is symmetric.
            # The current _apply_symmetry just copies.
            # If quarter is not symmetric (r,c) vs (c,r), the full cell will not be symmetric under transpose.
            # We should enforce symmetry in the search space or in _apply_symmetry.
            # Let's enforce in _apply_symmetry:
            # Actually, the problem states: cell == its transpose.
            # So we must ensure cell[r,c] == cell[c,r].
            # My _apply_symmetry sets:
            # cell[r,c] = val
            # cell[c,r] = val
            # So it forces transpose symmetry.
            # It also sets cell[r, size-1-c] = val, etc.
            # So it is correct.
            pass
    
    return cell

def _compute_gap(cell, n_pw=13, n_bands=6, k_points=None):
    """Compute the gap between bands 4 and 5 for TM polarization."""
    if k_points is None:
        k_points = [
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (0.25, 0.0),
            (0.75, 0.0),
            (0.5, 0.25),
            (0.5, 0.75),
            (0.25, 0.25),
            (0.75, 0.75),
            (0.25, 0.75),
            (0.75, 0.25),
            (0.5, 0.125),
            (0.125, 0.5)
        ]
        
    try:
        size = cell.shape[0]
        eps_high = 12.0
        eps_low = 1.0
        
        inv_eps = np.where(cell == 1, 1.0 / eps_high, 1.0 / eps_low)
        fft_2d = np.fft.fft2(inv_eps)
        
        n = n_pw
        n_half = n // 2
        G_indices = np.arange(-n_half, n_half + 1)
        
        # Precompute f_G
        f_G = np.zeros((n, n), dtype=complex)
        for i, m in enumerate(G_indices):
            for j, ng in enumerate(G_indices):
                fft_idx_i = m % size
                fft_idx_j = ng % size
                f_G[i, j] = fft_2d[fft_idx_i, fft_idx_j] / (size * size)
        
        # Precompute G vectors
        G_list = []
        for i in range(n):
            for j in range(n):
                gx = G_indices[i] * 2.0 * np.pi
                gy = G_indices[j] * 2.0 * np.pi
                G_list.append((gx, gy))
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            k_actual = np.array([kx * 2.0 * np.pi, ky * 2.0 * np.pi])
            
            H = np.zeros((n*n, n*n), dtype=complex)
            
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
                        idx_m = mg + n_half
                        idx_n = ng + n_half
                        f_val = f_G[idx_m, idx_n]
                    else:
                        f_val = 0.0
                    
                    H[a, b] = kp2 * f_val
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(np.real(eigenvalues))
            except Exception:
                continue
            
            if len(eigenvalues) >= n_bands:
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
    except Exception:
        return -1.0

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses local search to optimize the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem["pixels"]
    half = size // 2  # 16
    
    # Start with Dielectric Background (1s) and Air Holes (0s)
    # This is the standard configuration for large TM gaps.
    cell = np.ones((size, size), dtype=int)
    
    # Initial guess: Central hole and 4 corner holes
    # Center is around (15,15), (15,16), (16,15), (16,16)
    # Let's make a hole of radius ~3 pixels at center
    # And holes at (8,8), (8,23), (23,8), (23,23)
    
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    # Central hole
    R_central = 0.15
    dist_c = np.sqrt(X**2 + Y**2)
    cell[dist_c <= R_central] = 0
    
    # Corner holes
    dx = 0.25
    dy = 0.25
    R_corner = 0.08
    dist1 = np.sqrt((X - dx)**2 + (Y - dy)**2)
    cell[dist1 <= R_corner] = 0
    dist2 = np.sqrt((X + dx)**2 + (Y - dy)**2)
    cell[dist2 <= R_corner] = 0
    dist3 = np.sqrt((X - dx)**2 + (Y + dy)**2)
    cell[dist3 <= R_corner] = 0
    dist4 = np.sqrt((X + dx)**2 + (Y + dy)**2)
    cell[dist4 <= R_corner] = 0
    
    # Extract quarter
    quarter = cell[:half, :half].copy()
    
    # Evaluate initial design
    # Use smaller n_pw for speed during search
    n_pw_search = 11
    best_score = _compute_gap(cell, n_pw=n_pw_search)
    best_quarter = quarter.copy()
    
    # Local search
    # We will iterate through pixels in the quarter and try flipping them.
    # To be efficient, we can do multiple passes.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle positions to avoid bias? Or just iterate.
        # Let's iterate row by row.
        for r in range(half):
            for c in range(half):
                if time.time() > deadline:
                    break
                
                # Try flipping pixel (r, c)
                new_quarter = quarter.copy()
                new_quarter[r, c] = 1 - new_quarter[r, c]
                
                new_cell = _apply_symmetry(new_quarter, size)
                
                score = _compute_gap(new_cell, n_pw=n_pw_search)
                
                if score > best_score:
                    best_score = score
                    best_quarter = new_quarter.copy()
                    quarter = new_quarter
                    improved = True
    
    # Final evaluation with higher accuracy if time permits
    final_cell = _apply_symmetry(best_quarter, size)
    
    # Check if we have time for a more accurate check
    if time.time() < deadline - 5:
        final_score = _compute_gap(final_cell, n_pw=13)
        # Note: final_score is not used for return, but we could use it to decide if we should restart?
        # For now, just return the best found.
    
    return final_cell.tolist()
