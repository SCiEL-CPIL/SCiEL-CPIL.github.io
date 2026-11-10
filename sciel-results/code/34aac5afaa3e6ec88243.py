import numpy as np
import scipy.optimize as optimize
import time

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    Uses vectorized operations.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # Pairwise sums S_ij = s_i + s_j
    S = s[:, None] + s[None, :]
    # Thresholds T_kl = 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]
    # Pairwise weights W_ij = w_i * w_j
    W = w[:, None] * w[None, :]
    
    # Flatten for sorting and search
    S_flat = S.flatten()
    T_flat = T.flatten()
    W_flat = W.flatten()
    
    # Sort S_flat and compute cumulative weights
    order_S = np.argsort(S_flat, kind='stable')
    S_sorted = S_flat[order_S]
    W_sorted_S = W_flat[order_S]
    cum_W_S = np.cumsum(W_sorted_S)
    
    # For each T_flat, find how many S_sorted are strictly less than T_flat
    # searchsorted with side='left' gives the index of the first element >= T_flat
    # So the number of elements < T_flat is the index itself.
    # The sum of weights of those elements is cum_W_S[index - 1] (if index > 0) else 0.
    # We can use padded cumsum to handle index 0.
    
    idxs_F = np.searchsorted(S_sorted, T_flat, side='left')
    
    # Create padded cumulative sum: [0, cum_W_S[0], cum_W_S[1], ...]
    # If idx is k, we want sum of first k weights.
    # padded_cum[k] = sum of weights at indices 0..k-1 in sorted order.
    padded_cum_W_S = np.concatenate(([0.0], cum_W_S))
    
    # F_flat[i] = sum of W for S < T_i
    F_flat = padded_cum_W_S[idxs_F]
    F = F_flat.reshape(n, n)
    
    P = np.sum(W * F)
    return float(P)

def _validate_and_normalize(s, w):
    s = np.asarray(s, dtype=np.float64).flatten()
    w = np.asarray(w, dtype=np.float64).flatten()
    
    # Remove NaN/Inf
    mask = np.isfinite(s) & np.isfinite(w)
    s = s[mask]
    w = w[mask]
    
    if len(s) == 0:
        return None, None
        
    # Ensure s >= 0 (task says s_i >= -1e-9, so we clip)
    s = np.maximum(s, 0.0)
    
    # Ensure w >= 0
    w = np.maximum(w, 0.0)
    
    total_w = w.sum()
    if total_w <= 1e-9:
        return None, None
        
    # Sort by support to ensure consistency (optional but good)
    sort_idx = np.argsort(s)
    s = s[sort_idx]
    w = w[sort_idx]
    
    # Merge duplicate support points?
    # The task doesn't forbid duplicates, but it's inefficient.
    # However, if we have duplicates, we can just sum weights.
    unique_s, unique_indices, counts = np.unique(s, return_inverse=True, return_counts=True)
    if len(unique_s) < len(s):
        w_unique = np.zeros(len(unique_s))
        np.add.at(w_unique, unique_indices, w)
        s = unique_s
        w = w_unique
        total_w = w.sum()
        if total_w <= 1e-9:
            return None, None
        w = w / total_w
    else:
        w = w / total_w
        
    return s, w

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])
    best_score = _compute_P_fast(best_support, best_weights)
    
    # Helper to update best
    def try_update(s, w):
        nonlocal best_support, best_weights, best_score
        if time.time() > deadline:
            return
        s, w = _validate_and_normalize(s, w)
        if s is None:
            return
        score = _compute_P_fast(s, w)
        if score > best_score:
            best_score = score
            best_support = s
            best_weights = w

    # 1. Binary {0, 1}
    # P(p) = p^2(1-p)(3-2p)
    # Optimize p in [0, 1]
    def obj_binary(p):
        return -p**2 * (1-p) * (3-2*p)
    
    res = optimize.minimize_scalar(obj_binary, bounds=(0, 1), method='bounded')
    p_opt = res.x
    s_bin = np.array([0.0, 1.0])
    w_bin = np.array([p_opt, 1-p_opt])
    try_update(s_bin, w_bin)
    
    # 2. Mixture p * delta_0 + (1-p) * U[0, 1]
    # Approximate U[0,1] with grid
    n_grid = 100
    def eval_mixture_uniform(p, A=1.0):
        # Support: 0, and grid on [0, A]
        # Weights: p + (1-p)/n_grid for 0? No, 0 is a point mass.
        # The uniform part is on [0, A].
        # Let's define support as [0] U grid(0, A, n_grid)
        # But 0 is in both.
        # Weight at 0: p + (1-p) * (1/n_grid)
        # Weight at other grid points: (1-p) * (1/n_grid)
        
        grid = np.linspace(0, A, n_grid)
        # Ensure 0 is unique in support
        # grid[0] is 0.
        
        w_grid = np.full(n_grid, (1-p)/n_grid)
        w_grid[0] += p
        
        return grid, w_grid
    
    def obj_mixture_uniform(p):
        s, w = eval_mixture_uniform(p, 1.0)
        return -_compute_P_fast(s, w)
        
    res = optimize.minimize_scalar(obj_mixture_uniform, bounds=(0, 1), method='bounded')
    p_opt = res.x
    s_m, w_m = eval_mixture_uniform(p_opt, 1.0)
    try_update(s_m, w_m)
    
    # 3. Mixture p * delta_0 + (1-p) * U[0, A]
    # Optimize p and A
    def obj_mixture_uniform_A(params):
        p, A = params
        if p < 0 or p > 1 or A <= 0:
            return 1.0 # Invalid
        s, w = eval_mixture_uniform(p, A)
        return -_compute_P_fast(s, w)
        
    # Grid search for A first to guide optimizer? Or just use Nelder-Mead
    # Let's try a few A values and optimize p for each
    for A in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        def obj_p_for_A(p):
            s, w = eval_mixture_uniform(p, A)
            return -_compute_P_fast(s, w)
            
        res = optimize.minimize_scalar(obj_p_for_A, bounds=(0, 1), method='bounded')
        p_opt = res.x
        s_m, w_m = eval_mixture_uniform(p_opt, A)
        try_update(s_m, w_m)
        
    # 4. Mixture p0*delta_0 + p1*delta_1 + p2*delta_A
    # 3 points. Optimize p0, p1, A (p2 = 1-p0-p1)
    # This is a 3D optimization.
    # Let's do a coarse grid search for A and p1, then optimize p0?
    # Or just Nelder-Mead on (p0, p1, A)
    
    def eval_3point(p0, p1, A):
        p2 = 1 - p0 - p1
        if p0 < 0 or p1 < 0 or p2 < 0:
            return None, None
        s = np.array([0.0, 1.0, A])
        w = np.array([p0, p1, p2])
        return s, w
        
    def obj_3point(params):
        p0, p1, A = params
        if p0 < 0 or p1 < 0 or p0 + p1 > 1 or A <= 0:
            return 1.0
        s, w = eval_3point(p0, p1, A)
        if s is None:
            return 1.0
        return -_compute_P_fast(s, w)
        
    # Start from a reasonable guess
    x0 = [0.5, 0.2, 2.0]
    try:
        res = optimize.minimize(obj_3point, x0, method='Nelder-Mead', 
                                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        p0, p1, A = res.x
        s_m, w_m = eval_3point(p0, p1, A)
        if s_m is not None:
            try_update(s_m, w_m)
    except Exception:
        pass
        
    # 5. Mixture p0*delta_0 + p1*delta_1 + (1-p0-p1)*U[0, A]
    # This is more complex.
    
    # Final check and return
    # Ensure arrays are float64
    return best_support.astype(np.float64), best_weights.astype(np.float64)
