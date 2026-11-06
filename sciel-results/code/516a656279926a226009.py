import numpy as np
import time

def _enforce_symmetry(cell):
    """Enforce C4v symmetry on a 32x32 cell."""
    n = len(cell)
    cell = np.array(cell, dtype=np.int8)
    sym_sum = np.zeros_like(cell, dtype=np.float64)
    sym_sum += cell
    sym_sum += cell[::-1, ::-1]
    sym_sum += cell[::-1, :]
    sym_sum += cell[:, ::-1]
    sym_sum += cell.T
    sym_sum += cell[:, ::-1].T
    sym_sum += cell[::-1, :].T
    sym_sum += cell[::-1, ::-1].T
    cell = (sym_sum >= 4).astype(np.int8)
    return cell

def _compute_inv_eps_fourier(cell, n_pw_half=8):
    """Compute Fourier coefficients of 1/epsilon."""
    n = len(cell)
    eps = np.array(cell, dtype=np.float64)
    eps[eps == 1] = 12.0
    eps[eps == 0] = 1.0
    inv_eps = 1.0 / eps
    
    G_vals = np.arange(-n_pw_half, n_pw_half + 1)
    n_g = len(G_vals)
    coeffs = np.zeros((n_g, n_g), dtype=np.complex128)
    
    x = np.arange(n, dtype=np.float64)
    for gi, gx in enumerate(G_vals):
        for gj, gy in enumerate(G_vals):
            phase = np.exp(-1j * 2 * np.pi * (gx * x[:, None] + gy * x[None, :]) / n)
            coeffs[gi, gj] = np.sum(inv_eps * phase) / (n * n)
    
    return coeffs, G_vals

def _solve_tm_bands(coeffs, G_vals, kx, ky, n_bands=6):
    """Solve TM band structure at k-point."""
    n_g = len(G_vals)
    # G vectors
    Gx = G_vals[:, None] * np.ones((1, n_g))
    Gy = np.ones((n_g, 1)) * G_vals[None, :]
    
    # (k+G) vectors
    kxG = kx + Gx
    kyG = ky + Gy
    
    # |k+G|^2
    kG_sq = kxG ** 2 + kyG ** 2
    
    # Build the generalized eigenvalue problem
    # For TM: sum_G' (k+G)·(k+G') / eps_{G-G'} c_{G'} = omega^2 c_G
    # This is: A c = omega^2 c where A_{G,G'} = (k+G)·(k+G') * eps_{G-G'}^{-1}... 
    # Actually the standard form is:
    # sum_G' (k+G)·(k+G') * (1/eps)_{G-G'} c_{G'} = omega^2 c_G
    
    # (1/eps)_{G-G'} = coeffs corresponding to G-G'
    # G_vals is indexed, so G-G' has index diff = gi - gj
    # But coeffs is indexed by G_vals values, not indices
    # Let me map: coeffs[gi, gj] = (1/eps)_{G_vals[gi] - G_vals[gj]}? No.
    # coeffs[gi, gj] is the Fourier coefficient at (G_vals[gi], G_vals[gj])
    # So (1/eps)_{G} where G = (gx, gy) is coeffs[idx_gx, idx_gy]
    
    # For the matrix element A_{(ix,iy),(jx,jy)}:
    # G_diff = (G_vals[ix]-G_vals[jx], G_vals[iy]-G_vals[jy])
    # Need coeffs at that G_diff
    
    mat = np.zeros((n_g * n_g, n_g * n_g), dtype=np.complex128)
    
    for ix in range(n_g):
        for iy in range(n_g):
            idx1 = iy * n_g + ix
            for jx in range(n_g):
                for jy in range(n_g):
                    idx2 = jy * n_g + jx
                    # G_diff
                    dx = G_vals[ix] - G_vals[jx]
                    dy = G_vals[iy] - G_vals[jy]
                    # Find index in coeffs
                    ix_c = G_vals.tolist().index(dx) if dx in G_vals else None
                    iy_c = G_vals.tolist().index(dy) if dy in G_vals else None
                    if ix_c is None or iy_c is None:
                        continue
                    eps_coeff = coeffs[ix_c, iy_c]
                    if abs(eps_coeff) < 1e-12:
                        continue
                    # (k+G1)·(k+G2)
                    dot = (kx + G_vals[ix]) * (kx + G_vals[jx]) + (ky + G_vals[iy]) * (ky + G_vals[jy])
                    mat[idx1, idx2] = dot / eps_coeff
    
    # Solve eigenvalue problem
    try:
        eigvals, eigvecs = np.linalg.eig(mat)
        # Take real parts and sort
        eigvals = np.real(eigvals)
        mask = eigvals > 0
        eigvals = eigvals[mask]
        eigvals = np.sort(eigvals)
        return eigvals[:n_bands]
    except:
        return np.zeros(n_bands)

def _compute_gap(cell, n_pw_half=6, n_k=6):
    """Compute the gap between bands 4 and 5."""
    coeffs, G_vals = _compute_inv_eps_fourier(cell, n_pw_half)
    
    # K-points in irreducible BZ: (0,0), (0.5,0), (0.5,0.5)
    k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
    
    max_band4 = 0.0
    min_band5 = float('inf')
    
    for kx, ky in k_points:
        bands = _solve_tm_bands(coeffs, G_vals, kx, ky, n_bands=6)
        if len(bands) >= 5:
            max_band4 = max(max_band4, bands[3])  # 4th band (0-indexed: 3)
            min_band5 = min(min_band5, bands[4])  # 5th band (0-indexed: 4)
    
    if min_band5 == float('inf') or max_band4 == 0.0:
        return 0.0
    
    gap = 2 * (min_band5 - max_band4) / (min_band5 + max_band4)
    return gap

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 220
    
    size = problem["pixels"]
    
    # Start with a dielectric background with central air hole
    cell = np.ones((size, size), dtype=np.int8)
    center = size / 2.0
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5) / size - 0.5
            dy = (j + 0.5) / size - 0.5
            r = np.sqrt(dx * dx + dy * dy)
            if r < 0.12:
                cell[i, j] = 0
    
    cell = _enforce_symmetry(cell)
    best_cell = cell.copy()
    best_score = _compute_gap(best_cell)
    
    # Local search
    fund_pixels = []
    for i in range(size):
        for j in range(size):
            if i >= j and i >= size - 1 - j:
                fund_pixels.append((i, j))
    
    np.random.seed(42)
    
    while time.time() < deadline:
        idx = np.random.randint(len(fund_pixels))
        i, j = fund_pixels[idx]
        
        new_cell = best_cell.copy()
        positions = [(i, j), (j, i), (size-1-i, j), (i, size-1-j),
                     (size-1-j, size-1-i), (size-1-i, size-1-j),
                     (j, size-1-i), (size-1-j, i)]
        positions = list(set(positions))
        
        old_val = new_cell[i, j]
        for pi, pj in positions:
            new_cell[pi, pj] = 1 - old_val
        
        new_cell = _enforce_symmetry(new_cell)
        score = _compute_gap(new_cell)
        
        if score > best_score:
            best_cell = new_cell
            best_score = score
    
    return best_cell.tolist()
