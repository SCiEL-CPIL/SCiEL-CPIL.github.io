# EVOLVE-BLOCK-START
import numpy as np
import time


def _mc_volume_from_bases(base_x, base_y, N=8, n_mc=500000):
    """Compute the MC volume estimate directly from base positions."""
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()
    
    pos_x = base_x[i_flat]
    pos_y = base_y[j_flat]
    
    X0 = min(base_x[i] for i in range(N))
    X1 = max(base_x[i] + (i + 1) / N for i in range(N))
    Y0 = min(base_y[j] for j in range(N))
    Y1 = max(base_y[j] + (j + 1) / N for j in range(N))
    
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return float('inf')
    
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, n_mc)
    b = rng.uniform(Y0, Y1, n_mc)
    c = rng.uniform(0, 1, n_mc)
    
    inside = np.zeros(n_mc, dtype=bool)
    for k in range(N * N):
        i_k = i_flat[k]
        j_k = j_flat[k]
        x_lo = pos_x[k] + c * i_k / N
        x_hi = x_lo + 1.0 / N
        y_lo = pos_y[k] + c * j_k / N
        y_hi = y_lo + 1.0 / N
        mask = (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
        inside |= mask
    
    frac = np.mean(inside)
    V = frac * box_area
    if not np.isfinite(V) or V <= 0:
        return float('inf')
    return V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses local search to minimize the Monte Carlo union volume estimate.
    """
    N = 8
    deadline = time.time() + 620.0
    
    def make_positions(base_x, base_y):
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        i_flat = i_grid.ravel()
        j_flat = j_grid.ravel()
        pos_x = base_x[i_flat]
        pos_y = base_y[j_flat]
        return np.stack([pos_x, pos_y], axis=1)
    
    # Keich initialization
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    best_x = keich.copy()
    best_y = keich.copy()
    best_V = _mc_volume_from_bases(best_x, best_y, N)
    
    # Local search
    def local_search(base_x, base_y, max_iters=2000):
        curr_x = base_x.copy()
        curr_y = base_y.copy()
        curr_V = _mc_volume_from_bases(curr_x, curr_y, N)
        
        step_sizes = [0.03, 0.015, 0.008, 0.004, 0.002]
        no_improve = 0
        max_no_improve = 150
        
        for it in range(max_iters):
            if time.time() > deadline - 40:
                break
            if no_improve > max_no_improve:
                break
            
            dim = np.random.randint(2)
            idx = np.random.randint(N)
            step = np.random.choice(step_sizes) * np.random.choice([-1, 1])
            
            if dim == 0:
                new_x = curr_x.copy()
                new_x[idx] += step
                new_y = curr_y
            else:
                new_x = curr_x
                new_y = curr_y.copy()
                new_y[idx] += step
            
            new_V = _mc_volume_from_bases(new_x, new_y, N)
            if new_V < curr_V:
                curr_x, curr_y, curr_V = new_x, new_y, new_V
                no_improve = 0
            else:
                no_improve += 1
        
        return curr_x, curr_y, curr_V
    
    # Multi-start
    n_starts = 15
    for start in range(n_starts):
        if time.time() > deadline - 80:
            break
        
        if start == 0:
            init_x = best_x.copy()
            init_y = best_y.copy()
        else:
            rng_local = np.random.default_rng(start * 7919 + 13)
            init_x = best_x + rng_local.normal(0, 0.08, N)
            init_y = best_y + rng_local.normal(0, 0.08, N)
        
        result_x, result_y, result_V = local_search(init_x, init_y, max_iters=1500)
        
        if result_V < best_V:
            best_x, best_y, best_V = result_x, result_y, result_V
    
    positions = make_positions(best_x, best_y)
    
    # Final check
    final_V = _mc_volume_from_bases(best_x, best_y, N)
    if not np.isfinite(final_V) or final_V <= 0:
        return make_positions(keich, keich)
    
    return positions


# EVOLVE-BLOCK-END
