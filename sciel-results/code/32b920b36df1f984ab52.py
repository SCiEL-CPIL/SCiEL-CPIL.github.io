import numpy as np
from scipy.optimize import minimize
import time

def _compute_1d_union_length(base_pos, c_samples):
    """
    Compute the integral of the length of the union of intervals for given base positions.
    base_pos: array of shape (N,)
    c_samples: array of shape (M,)
    Returns: approximate integral value.
    """
    N = len(base_pos)
    M = len(c_samples)
    # Intervals at each c are [x_i + c*i/N, x_i + 1/N + c*i/N]
    # We can compute the union length for each c.
    
    # To make it vectorized, we can sort the intervals and merge them.
    # However, for N=8, a simple loop over c_samples is fast enough if M is moderate.
    
    lengths = np.zeros(M)
    for k in range(M):
        c = c_samples[k]
        starts = base_pos + c * np.arange(N) / N
        ends = starts + 1.0 / N
        # Sort by start
        idx = np.argsort(starts)
        s = starts[idx]
        e = ends[idx]
        
        # Merge intervals
        current_end = e[0]
        total_len = 0.0
        for j in range(1, N):
            if s[j] > current_end:
                total_len += current_end - s[j-1] if j > 1 else current_end - s[0]
                # Actually, standard merge:
                # Let's do it properly
                pass
        
        # Simpler merge:
        merged_starts = [s[0]]
        merged_ends = [e[0]]
        for j in range(1, N):
            if s[j] <= merged_ends[-1]:
                merged_ends[-1] = max(merged_ends[-1], e[j])
            else:
                merged_starts.append(s[j])
                merged_ends.append(e[j])
        
        total_len = sum(e - s for s, e in zip(merged_starts, merged_ends))
        lengths[k] = total_len
        
    return np.trapz(lengths, c_samples)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 580  # Leave 20s margin
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    x0 = keich.copy()
    
    # Define objective function for 1D optimization
    # We sample c in [0, 1] to approximate the integral
    c_samples = np.linspace(0, 1, 200)
    
    def objective(base_pos):
        return _compute_1d_union_length(base_pos, c_samples)
    
    # Optimize
    best_x = x0.copy()
    best_val = objective(best_x)
    
    # Try Nelder-Mead
    try:
        res = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 1000, 'fatol': 1e-6})
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
    else:
        # Try a few random restarts if time permits
        rng = np.random.default_rng(12345)
        for _ in range(10):
            if time.time() > deadline:
                break
            x_rand = rng.uniform(-1, 1, N)
            try:
                res = minimize(objective, x_rand, method='Nelder-Mead', options={'maxiter': 500, 'fatol': 1e-6})
                if res.success and res.fun < best_val:
                    best_x = res.x
                    best_val = res.fun
            except Exception:
                pass
    
    # Construct 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions

