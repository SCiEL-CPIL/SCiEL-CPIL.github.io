"""Photonic crystal band gap optimizer with fast PW method and simulated annealing."""
import numpy as np
import time

def _tm_bands_fast(cell_arr, kx, ky, n_pw=7):
    """Compute TM band frequencies using fast PW method with FFT."""
    size = len(cell_arr)
    eps_high = 12.0
    eps_low = 1.0
    
    inv_eps = np.where(cell_arr == 1, 1.0/eps_high, 1.0/eps_low)
    fft_2d = np.fft.fft2(inv_eps)
    
    half = n_pw // 2
    if n_pw % 2 == 1:
        G_vals = np.arange(-half, half + 1)
    else:
        G_vals = np.arange(-n_pw//2, n_pw//2 + 1)
    
    nG = len(G_vals)
    total = nG * nG
    
    kG_x = np.zeros(total)
    kG_y = np.zeros(total)
    G_idx = []
    
    for gi in range(nG):
        for gj in range(nG):
            idx = gi * nG + gj
            kG_x[idx] = kx + G_vals[gi]
            kG_y[idx] = ky + G_vals[gj]
            G_idx.append((G_vals[gi], G_vals[gj]))
    
    H = np.zeros((total, total), dtype=float)
    
    for i in range(total):
        for j in range(total):
            Gx_i, Gy_i = G_idx[i]
            Gx_j, Gy_j = G_idx[j]
            Gdiff_x = Gx_i - Gx_j
            Gdiff_y = Gy_i - Gy_j
            fx = Gdiff_x % size
            fy = Gdiff_y % size
            eps_coeff = fft_2d[fy, fx] / (size * size)
            if abs(eps_coeff) < 1e-15:
                continue
            dot = kG_x[i] * kG_x[j] + kG_y[i] * kG_y[j]
            H[i, j] = dot / eps_coeff.real
    
    try:
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.real(eigvals)
        eigvals = np.clip(eigvals, 0, None)
        freqs = np.sqrt(eigvals)
        return np.sort(freqs)
    except:
        return np.array([0.0] * total)

def _gap_ratio_fast(cell, n_pw=7):
    cell_arr = np.array(cell, dtype=int)
    k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.25), (0.3, 0.1), (0.4, 0.2)]
    max_omega4 = 0.0
    min_omega5 = np.inf
    for kx, ky in k_points:
        freqs = _tm_bands_fast(cell_arr, kx, ky, n_pw)
        if len(freqs) >= 5:
            max_omega4 = max(max_omega4, freqs[3])
            min_omega5 = min(min_omega5, freqs[4])
    if min_omega5 == np.inf or min_omega5 <= 0 or max_omega4 <= 0:
        return -1.0
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap <= 0:
        return -1.0
    return gap / midgap

def _enforce_symmetry(cell):
    size = len(cell)
    result = [row[:] for row in cell]
    for i in range(size // 2):
        for j in range(size):
            val = 1 if result[i][j] + result[size-1-i][j] >= 1 else 0
            result[i][j] = val
            result[size-1-i][j] = val
    for i in range(size):
        for j in range(size // 2):
            val = 1 if result[i][j] + result[i][size-1-j] >= 1 else 0
            result[i][j] = val
            result[i][size-1-j] = val
    for i in range(size):
        for j in range(i + 1, size):
            val = 1 if result[i][j] + result[j][i] >= 1 else 0
            result[i][j] = val
            result[j][i] = val
    return result

def design(problem):
    start_time = time.time()
    deadline = start_time + 110
    size = problem["pixels"]
    
    # Multiple initial patterns
    cell1 = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (j+0.5)/size - 0.5
            y = (i+0.5)/size - 0.5
            if 0.15 < np.sqrt(x*x+y*y) < 0.35:
                cell1[i][j] = 1
    cell1 = _enforce_symmetry(cell1)
    
    cell2 = [[0]*size for _ in range(size)]
    mid = size//2
    for i in range(size):
        for j in range(size):
            if abs(i-mid+0.5)<3 or abs(j-mid+0.5)<3:
                cell2[i][j] = 1
    cell2 = _enforce_symmetry(cell2)
    
    candidates = [cell1, cell2]
    best_cell = candidates[0]
    best_score = _gap_ratio_fast(candidates[0])
    for c in candidates[1:]:
        s = _gap_ratio_fast(c)
        if s > best_score:
            best_score = s
            best_cell = c
    
    cell = [row[:] for row in best_cell]
    current_score = best_score
    temp = 0.05
    cooling = 0.998
    np.random.seed(42)
    
    while time.time() < deadline:
        new_cell = [row[:] for row in cell]
        for _ in range(np.random.randint(1, 4)):
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            new_cell[i][j] = 1 - new_cell[i][j]
        new_cell = _enforce_symmetry(new_cell)
        new_score = _gap_ratio_fast(new_cell)
        if new_score > best_score:
            best_cell = [row[:] for row in new_cell]
            best_score = new_score
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / max(temp, 1e-10)):
            cell = new_cell
            current_score = new_score
        temp = max(temp * cooling, 0.005)
    
    return best_cell
