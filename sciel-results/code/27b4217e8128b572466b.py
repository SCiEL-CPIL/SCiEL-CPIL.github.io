# EVOLVE-BLOCK-START
import numpy as np
import time

def _union_length_1d(p, N, n_c=2001):
    """Compute integral of 1D union length over c in [0,1]."""
    idx = np.arange(N)
    c = np.linspace(0, 1, n_c)
    left = p[:, None] + c[None, :] * idx[:, None] / N
    right = p[:, None] + (1 + idx[:, None]) / N + c[None, :] * idx[:, None] / N
    order = np.argsort(left, axis=0)
    sorted_left = np.take_along_axis(left, order, axis=0)
    sorted_right = np.take_along_axis(right, order, axis=0)
    n_c_local = len(c)
    union_len = np.zeros(n_c_local)
    for t in range(n_c_local):
        sl = sorted_left[:, t]
        sr = sorted_right[:, t]
        cur_end = sl[0]
        cur_start = sl[0]
        total = 0.0
        for i in range(1, N):
            if sl[i] > cur_end:
                total += cur_end - cur_start
                cur_start = sl[i]
                cur_end = sr[i]
            else:
                cur_end = max(cur_end, sr[i])
        total += cur_end - cur_start
        union_len[t] = total
    return np.trapz(union_len, c)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_p = keich.copy()
    best_val = _union_length_1d(best_p, N, 2001)
    
    # Simple local search / coordinate descent with perturbations
    rng = np.random.default_rng(42)
    step_size = 0.05
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 30 and no_improve < max_no_improve:
        # Try random perturbations
        p_try = best_p.copy()
        idx = rng.integers(0, N)
        p_try[idx] += rng.normal(0, step_size)
        
        val = _union_length_1d(p_try, N, 1001)
        if val < best_val - 1e-12:
            best_p = p_try.copy()
            best_val = val
            no_improve = 0
            # Try to reduce step size occasionally
            if no_improve == 0 and best_val < 0.3:
                step_size = max(0.005, step_size * 0.9)
        else:
            no_improve += 1
    
    # Final polish with smaller steps
    if time.time() < deadline - 10:
        step_size = 0.005
        no_improve = 0
        while time.time() < deadline - 5 and no_improve < 50:
            p_try = best_p.copy()
            idx = rng.integers(0, N)
            p_try[idx] += rng.normal(0, step_size)
            val = _union_length_1d(p_try, N, 1001)
            if val < best_val - 1e-12:
                best_p = p_try.copy()
                best_val = val
                no_improve = 0
            else:
                no_improve += 1
    
    # Build final positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = best_p[i_flat]
    base_y = best_p[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
