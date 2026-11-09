"""Design a 32x32 photonic crystal unit cell maximizing TM band gap 4-5."""
import time
import numpy as np
from scipy.linalg import eigh

def design(problem):
    start_time = time.time()
    deadline = start_time + 215
    
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Fast plane-wave basis: 7x7 = 49 plane waves
    Npw = 7
    half = Npw // 2  # 3
    G = np.array([(2.0 * np.pi * m, 2.0 * np.pi * n) 
                  for m in range(-half, half + 1) 
                  for n in range(-half, half + 1)])
    nG = len(G)
    
    # BZ points (irreducible)
    bz_points = []
    for t in np.linspace(0, 0.5, 5):
        bz_points.append((t, 0.0))
    for t in np.linspace(0, 0.5, 5):
        bz_points.append((0.5, t))
    for t in np.linspace(0, 0.5, 5):
        bz_points.append((t, t))
    bz_points = list(set([(round(x, 3), round(y, 3)) for x, y in bz_points]))
    bz_points.sort()
    bz_points = bz_points[:20]
    
    def compute_fft_inv_eps(pixel_cell):
        eps_map = np.where(np.array(pixel_cell, dtype=float) > 0.5, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        return np.fft.fft2(inv_eps) / (size * size)
    
    def solve_bands(fft_inv_eps, kx, ky, n_bands=6):
        k = np.array([kx, ky])
        kG = k + G
        M = np.zeros((nG, nG), dtype=complex)
        for i in range(nG):
            for j in range(nG):
                dm = int(G[i, 0] / (2.0 * np.pi)) - int(G[j, 0] / (2.0 * np.pi))
                dn = int(G[i, 1] / (2.0 * np.pi)) - int(G[j, 1] / (2.0 * np.pi))
                mi = dm % size
                ni = dn % size
                c = fft_inv_eps[mi, ni]
                M[i, j] = (kG[i, 0] * kG[j, 0] + kG[i, 1] * kG[j, 1]) * c
        try:
            eigvals = eigh(M, eigvals_only=True, subset_by_index=[0, n_bands - 1])
            return eigvals[:n_bands]
        except:
            return None
    
    def evaluate_gap(pixel_cell):
        fft_inv_eps = compute_fft_inv_eps(pixel_cell)
        max_omega4 = 0.0
        min_omega5 = np.inf
        for kx, ky in bz_points:
            eigvals = solve_bands(fft_inv_eps, kx, ky, n_bands=6)
            if eigvals is None or len(eigvals) < 5:
                continue
            omegas = np.sqrt(np.maximum(eigvals, 0))
            max_omega4 = max(max_omega4, omegas[3])
            min_omega5 = min(min_omega5, omegas[4])
        if min_omega5 == np.inf:
            return -1.0
        gap = 2.0 * (min_omega5 - max_omega4)
        midgap = min_omega5 + max_omega4
        if midgap <= 0:
            return -1.0
        return gap / midgap
    
    def make_symmetric_cell(params, n_terms=4):
        i_coords = np.linspace(-1, 1, 16)
        j_coords = np.linspace(-1, 1, 16)
        level = np.zeros((16, 16))
        for m in range(n_terms):
            for n in range(n_terms):
                idx = m * n_terms + n
                if idx >= len(params):
                    continue
                coeff = params[idx]
                if abs(coeff) < 1e-10:
                    continue
                for i_idx, x in enumerate(i_coords):
                    for j_idx, y in enumerate(j_coords):
                        tm_x = np.cos(m * np.arccos(np.clip(x, -1, 1)))
                        tn_y = np.cos(n * np.arccos(np.clip(y, -1, 1)))
                        tn_x = np.cos(n * np.arccos(np.clip(x, -1, 1)))
                        tm_y = np.cos(m * np.arccos(np.clip(y, -1, 1)))
                        val = 0.5 * (tm_x * tn_y + tn_x * tm_y)
                        level[i_idx, j_idx] += coeff * val
        fund16 = (level > 0).astype(int)
        fund16 = np.maximum(fund16, fund16.T)
        cell = np.zeros((32, 32), dtype=int)
        for i in range(16):
            for j in range(16):
                val = fund16[i, j]
                cell[i, j] = val
                cell[31 - i, j] = val
                cell[i, 31 - j] = val
                cell[31 - i, 31 - j] = val
        return cell.tolist()
    
    n_params = 16  # 4x4
    best_score = -np.inf
    best_params = np.zeros(n_params)
    best_cell = None
    
    # Initial: rod
    best_params[0] = 2.0
    initial_cell = make_symmetric_cell(best_params)
    best_score = evaluate_gap(initial_cell)
    best_cell = initial_cell
    
    rng = np.random.RandomState(42)
    
    n_starts = 3
    max_iter = 150
    
    for start_i in range(n_starts):
        if time.time() > deadline - 20:
            break
        if start_i == 0:
            params = best_params.copy()
        else:
            params = best_params + rng.randn(n_params) * 0.5
        current_score = evaluate_gap(make_symmetric_cell(params))
        
        for iteration in range(max_iter):
            if time.time() > deadline - 8:
                break
            step = 0.5 * (0.92 ** (iteration // 30))
            direction = rng.randn(n_params)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm
            new_params = params + step * direction
            new_cell = make_symmetric_cell(new_params)
            new_score = evaluate_gap(new_cell)
            if new_score > current_score:
                params = new_params
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_params = params.copy()
                    best_cell = new_cell
            else:
                new_params = params - step * direction
                new_cell = make_symmetric_cell(new_params)
                new_score = evaluate_gap(new_cell)
                if new_score > current_score:
                    params = new_params
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_params = params.copy()
                        best_cell = new_cell
    
    return best_cell
