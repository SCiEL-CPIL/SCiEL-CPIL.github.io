import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w = w / w.sum()
    
    # Vectorized O(n^2 log n)
    s_pair = s[:, None] + s[None, :]
    w_pair = w[:, None] * w[None, :]
    
    flat_s = s_pair.ravel()
    flat_w = w_pair.ravel()
    
    order = np.argsort(flat_s, kind='mergesort')
    sorted_s = flat_s[order]
    sorted_w = flat_w[order]
    cum_w = np.cumsum(sorted_w)
    
    # targets[k, l] = 2*s[l] - s[k]
    targets = 2.0 * s[None, :] - s[:, None]
    flat_targets = targets.ravel()
    
    idx = np.searchsorted(sorted_s, flat_targets, side='left')
    
    P_X2X3 = np.zeros_like(flat_targets)
    mask = idx > 0
    P_X2X3[mask] = cum_w[idx[mask] - 1]
    
    P_X2X3 = P_X2X3.reshape(n, n)
    
    P = np.sum(w[:, None] * w[None, :] * P_X2X3)
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    # 1. Analytical 2-point solution
    # p = (15 - sqrt(33)) / 16
    p_opt = (15 - np.sqrt(33)) / 16.0
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_opt, 1.0 - p_opt])
    P2 = _compute_P(s2, w2)
    if P2 > best_P:
        best_P = P2
        best_support = s2.copy()
        best_weights = w2.copy()

    # 2. Optimize 3-point distribution
    # Support: [0, 1, M]. Weights: log-space
    def _unpack3(x):
        M = np.exp(x[0])
        lw = x[1:4]
        w = np.exp(lw - np.max(lw))
        w = w / w.sum()
        s = np.array([0.0, 1.0, M])
        return s, w

    def _neg_P3(x):
        if time.time() > deadline: return 1e6
        s, w = _unpack3(x)
        return -_compute_P(s, w)

    # Initial guess: M=10, weights close to optimal 2-point shifted
    x0_3 = np.array([np.log(10.0), np.log(p_opt), np.log(1.0-p_opt), 0.0]) 
    # Note: 3 weights, so x[1:4] are 3 values. 
    # My _unpack3 takes x[1:4] which is 3 values. Correct.
    
    try:
        res = minimize(_neg_P3, x0_3, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
        if res.fun < -best_P:
            s3, w3 = _unpack3(res.x)
            P3 = _compute_P(s3, w3)
            if P3 > best_P:
                best_P = P3
                best_support = s3.copy()
                best_weights = w3.copy()
    except Exception:
        pass

    # 3. Optimize 4-point distribution
    # Support: [0, 1, M1, M2]. Weights: log-space
    def _unpack4(x):
        M1 = np.exp(x[0])
        M2 = np.exp(x[1])
        lw = x[2:6]
        w = np.exp(lw - np.max(lw))
        w = w / w.sum()
        s = np.array([0.0, 1.0, M1, M2])
        # Ensure sorted? Not strictly required for _compute_P but good for stability
        idx = np.argsort(s)
        s = s[idx]
        w = w[idx]
        return s, w

    def _neg_P4(x):
        if time.time() > deadline: return 1e6
        s, w = _unpack4(x)
        return -_compute_P(s, w)

    x0_4 = np.array([np.log(5.0), np.log(20.0), np.log(p_opt), np.log(1.0-p_opt), 0.0, 0.0])
    
    if time.time() < deadline - 100:
        try:
            res = minimize(_neg_P4, x0_4, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < -best_P:
                s4, w4 = _unpack4(res.x)
                P4 = _compute_P(s4, w4)
                if P4 > best_P:
                    best_P = P4
                    best_support = s4.copy()
                    best_weights = w4.copy()
        except Exception:
            pass

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    best_support = np.abs(best_support)
    best_weights = np.abs(best_weights)
    best_weights = best_weights / best_weights.sum()
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)
