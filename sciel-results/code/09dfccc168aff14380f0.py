import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(s, w):
    """
    Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w.
    Uses O(n^2 log n) method.
    """
    n = len(s)
    if n == 0:
        return 0.0
        
    # Pairwise sums S_ij = s_i + s_j
    S = s[:, np.newaxis] + s[np.newaxis, :]
    S_flat = S.flatten()
    
    # Weights for pairs V_ij = w_i * w_j
    V = w[:, np.newaxis] * w[np.newaxis, :]
    V_flat = V.flatten()
    
    # Sort S and V
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    V_sorted = V_flat[sort_idx]
    
    # Cumulative sum of V
    V_cumsum = np.cumsum(V_sorted)
    V_cumsum_padded = np.concatenate(([0.0], V_cumsum))
    
    # Thresholds T_kl = 2*s_l - s_k
    # We need sum_{i,j} w_i w_j [s_i + s_j < 2*s_l - s_k]
    # This is P_inner[k,l]
    T = 2 * s[np.newaxis, :] - s[:, np.newaxis]
    T_flat = T.flatten()
    
    idx = np.searchsorted(S_sorted, T_flat, side='left')
    P_inner = V_cumsum_padded[idx]
    
    # P = sum_{k,l} w_k w_l P_inner[k,l]
    W_prod = w[:, np.newaxis] * w[np.newaxis, :]
    P = np.sum(W_prod * P_inner.reshape(n, n))
    
    return float(P)

def _neg_P_3point(params):
    """
    Objective for 3-point distribution {0, 1, 3}.
    params: [log_w0, log_w1, log_w3]
    """
    # Softmax to get weights
    w_raw = np.exp(params)
    w = w_raw / np.sum(w_raw)
    s = np.array([0.0, 1.0, 3.0])
    return -_compute_P(s, w)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # 1. Optimize 3-point distribution {0, 1, 3}
    # Initial guess based on manual search: q~0.35, p~0.25, r~0.4
    # w = [0.4, 0.25, 0.35]
    init_params = np.log([0.4, 0.25, 0.35])
    
    res = minimize(_neg_P_3point, init_params, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000})
    
    if res.success or res.fun < _neg_P_3point(init_params):
        w_opt = np.exp(res.x)
        w_opt = w_opt / np.sum(w_opt)
        s_opt = np.array([0.0, 1.0, 3.0])
        score = -res.fun
        if score > best_score:
            best_score = score
            best_support = s_opt
            best_weights = w_opt

    # 2. Try a few other 3-point structures
    for scale in [2.0, 5.0, 10.0]:
        s_test = np.array([0.0, 1.0, scale])
        def _obj_test(params, s=s_test):
            w_raw = np.exp(params)
            w = w_raw / np.sum(w_raw)
            return -_compute_P(s, w)
        
        res_t = minimize(_obj_test, init_params, method='Nelder-Mead', options={'maxiter': 1000})
        if res_t.success:
            w_opt_t = np.exp(res_t.x)
            w_opt_t = w_opt_t / np.sum(w_opt_t)
            score_t = -res_t.fun
            if score_t > best_score:
                best_score = score_t
                best_support = s_test
                best_weights = w_opt_t

    # 3. Generic optimization for n=4, 5
    for n in [4, 5]:
        if time.time() > deadline - 120:
            break
            
        # Initial guess: extend 3-point with a small weight
        if n == 4:
            s_init = np.array([0.0, 1.0, 3.0, 10.0])
            w_init = np.array([0.4, 0.25, 0.35, 0.0]) # 0 weight for new point
            # Perturb slightly to allow optimization
            w_init[3] = 0.01
            w_init = w_init / np.sum(w_init)
        else:
            s_init = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
            w_init = np.array([0.4, 0.25, 0.35, 0.0, 0.0])
            w_init[3] = 0.005
            w_init[4] = 0.005
            w_init = w_init / np.sum(w_init)
            
        # Optimize weights only, fix support
        def _obj_w(params, s=s_init):
            w_raw = np.exp(params)
            w = w_raw / np.sum(w_raw)
            return -_compute_P(s, w)
            
        init_w_params = np.log(w_init)
        res_w = minimize(_obj_w, init_w_params, method='Nelder-Mead', options={'maxiter': 2000})
        
        if res_w.success:
            w_opt_w = np.exp(res_w.x)
            w_opt_w = w_opt_w / np.sum(w_opt_w)
            score_w = -res_w.fun
            if score_w > best_score:
                best_score = score_w
                best_support = s_init
                best_weights = w_opt_w

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0, 3.0])
        best_weights = np.array([0.4, 0.25, 0.35])
        
    # Ensure valid output
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    best_weights = best_weights / np.sum(best_weights)
    
    return best_support, best_weights
