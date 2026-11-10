import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = 600.0 - 30.0  # 30s safety margin
    
    # The cross-section at height c is a product of 1D unions in x and y.
    # Union volume = integral_0^1 Lx(c) * Ly(c) dc
    # where Lx(c) = length of union of [x_i + c*i/N, x_i + c*i/N + 1/N] for i=0..N-1
    # Similarly for Ly with y_j.
    # By symmetry, optimal x_i = y_j for the same index.
    # So we need to find 1D positions p_0,...,p_7 minimizing integral of L(c)^2 dc
    # where L(c) = union length of [p_i + c*i/N, p_i + c*i/N + 1/N]
    
    def compute_1d_union_length(c_vals, positions, N=8):
        """Compute union length of intervals [pos_i + c*i/N, pos_i + c*i/N + 1/N] for each c."""
        # positions: (N,), c_vals: (M,)
        # For each c, we have N intervals. Compute union length.
        M = len(c_vals)
        # Intervals: for each i, [positions[i] + c*i/N, positions[i] + c*i/N + 1/N]
        # Shape: (M, N, 2)
        offsets = c_vals[:, None] * np.arange(N)[None, :] / N  # (M, N)
        left = positions[None, :] + offsets  # (M, N)
        right = left + 1.0 / N  # (M, N)
        
        # For each row, compute union length of intervals
        union_lengths = np.zeros(M)
        for m in range(M):
            l = left[m]
            r = right[m]
            # Sort by left endpoint
            order = np.argsort(l)
            sl = l[order]
            sr = r[order]
            # Merge intervals
            total = 0.0
            cur_l = sl[0]
            cur_r = sr[0]
            for k in range(1, N):
                if sl[k] > cur_r:
                    total += cur_r - cur_l
                    cur_l = sl[k]
                    cur_r = sr[k]
                else:
                    cur_r = max(cur_r, sr[k])
            total += cur_r - cur_l
            union_lengths[m] = total
        return union_lengths
    
    def objective_1d(positions, N=8, n_samples=200):
        """Objective: integral of L(c)^2 dc, approximated by trapezoidal rule."""
        c_vals = np.linspace(0, 1, n_samples)
        L = compute_1d_union_length(c_vals, positions, N)
        return np.trapz(L * L, c_vals)
    
    # Initial guess: Keich-style or simple spread
    # Try a few initial guesses
    best_pos = None
    best_obj = np.inf
    
    initial_guesses = [
        np.linspace(0, 1 - 1/N, N),  # spread out
        np.zeros(N),  # all at origin
        np.array([i * 0.1 for i in range(N)]),
        # Keich-inspired
        np.array([(i & 1) * 0.5 + (i & 2) * 0.25 + (i & 4) * 0.125 for i in range(N)]),
        np.array([np.random.uniform(0, 1) for _ in range(N)]),
        np.array([np.random.uniform(0, 1) for _ in range(N)]),
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective_1d, guess, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_pos = result.x.copy()
        except:
            pass
    
    if best_pos is None:
        best_pos = np.linspace(0, 1 - 1/N, N)
    
    # Refine with more samples
    def objective_1d_refined(positions, N=8, n_samples=500):
        c_vals = np.linspace(0, 1, n_samples)
        L = compute_1d_union_length(c_vals, positions, N)
        return np.trapz(L * L, c_vals)
    
    if time.time() < deadline:
        try:
            result = minimize(objective_1d_refined, best_pos, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_obj:
                best_obj = result.fun
                best_pos = result.x.copy()
        except:
            pass
    
    # Build the 64 positions: tube (i,j) has base (p_i, p_j)
    positions_1d = best_pos
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = positions_1d[i_flat]
    base_y = positions_1d[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
