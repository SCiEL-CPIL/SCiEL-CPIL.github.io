import numpy as np
from scipy.optimize import minimize
import time

def _compute_P_fast(s, w):
    """Fast exact P[X1+X2+X3 < 2*X4] using pair-sum sorting."""
    n = s.size
    if n <= 1:
        return 0.0
    # Build all n^2 pairwise sums (for X1+X2)
    pair_sums = s[:, None] + s[None, :]
    pair_weights = w[:, None] * w[None, :]
    
    # Sort pair sums
    flat_s = pair_sums.ravel()
    flat_w = pair_weights.ravel()
    order = np.argsort(flat_s)
    sorted_sums = flat_s[order]
    sorted_ws = flat_w[order]
    cumsum = np.cumsum(sorted_ws)
    
    # For each l, k: need sum of w_i*w_j where s_i+s_j < 2*s_l - s_k
    # Total = sum_k w_k * sum_l w_l * H(2*s_l - s_k)
    # where H(t) = sum of pair weights with pair_sum < t
    total = 0.0
    for k in range(n):
        thresholds = 2.0 * s - s[k]
        idx = np.searchsorted(sorted_sums, thresholds, side='left')
        H = np.zeros(n)
        mask = idx > 0
        H[mask] = cumsum[idx[mask] - 1]
        total += w[k] * np.sum(w * H)
    return float(total)

def _compute_P_exact(support, weights):
    """Original 4D computation for validation."""
    n = len(support)
    s1 = support[:, np.newaxis, np.newaxis, np.newaxis]
    s2 = support[np.newaxis, :, np.newaxis, np.newaxis]
    s3 = support[np.newaxis, np.newaxis, :, np.newaxis]
    s4 = support[np.newaxis, np.newaxis, np.newaxis, :]
    
    w1 = weights[:, np.newaxis, np.newaxis, np.newaxis]
    w2 = weights[np.newaxis, :, np.newaxis, np.newaxis]
    w3 = weights[np.newaxis, np.newaxis, :, np.newaxis]
    w4 = weights[np.newaxis, np.newaxis, np.newaxis, :]
    
    mask = (s1 + s2 + s3) < (2.0 * s4)
    p = np.sum(w1 * w2 * w3 * w4 * mask)
    return p

def _optimize_n(n, s0, w0, maxiter=3000, deadline=None):
    """Optimize n-point measure using L-BFGS-B on log-parameters."""
    def objective(params):
        if deadline is not None and time.time() > deadline:
            return 1.0  # Penalty
        raw_s = params[:n]
        raw_w = params[n:]
        
        # Clip to prevent overflow
        raw_s = np.clip(raw_s, -20, 20)
        raw_w = np.clip(raw_w, -20, 20)
        
        s = np.exp(raw_s)
        w_unnorm = np.exp(raw_w - np.max(raw_w))
        w_sum = np.sum(w_unnorm)
        if w_sum <= 0 or not np.isfinite(w_sum):
            return 1.0
        w = w_unnorm / w_sum
        
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
            return 1.0
        
        p = _compute_P_fast(s, w)
        if not np.isfinite(p):
            return 1.0
        return -p
    
    log_s0 = np.log(np.maximum(s0, 1e-12))
    log_w0 = np.log(np.maximum(w0, 1e-12))
    x0 = np.concatenate([log_s0, log_w0])
    
    bounds = [(-20.0, 20.0)] * (2 * n)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': False})
        
        if res.x is not None and np.all(np.isfinite(res.x)):
            raw_s = np.clip(res.x[:n], -20, 20)
            raw_w = np.clip(res.x[n:], -20, 20)
            s = np.exp(raw_s)
            w_unnorm = np.exp(raw_w - np.max(raw_w))
            w_sum = np.sum(w_unnorm)
            if w_sum > 0 and np.isfinite(w_sum):
                w = w_unnorm / w_sum
                p = _compute_P_fast(s, w)
                if np.isfinite(p):
                    return s, w, p
    except Exception:
        pass
    
    return None, None, -1.0

def _optimize_n_nm(n, s0, w0, maxiter=2000, deadline=None):
    """Polish with Nelder-Mead."""
    def objective(params):
        if deadline is not None and time.time() > deadline:
            return 1.0
        v = params[:n]
        u = params[n:]
        
        v = np.clip(v, -20, 20)
        u = np.clip(u, -20, 20)
        
        s = np.exp(v)
        w_unnorm = np.exp(u - np.max(u))
        w_sum = np.sum(w_unnorm)
        if w_sum <= 0 or not np.isfinite(w_sum):
            return 1.0
        w = w_unnorm / w_sum
        
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
            return 1.0
        
        p = _compute_P_fast(s, w)
        if not np.isfinite(p):
            return 1.0
        return -p
    
    v0 = np.log(np.maximum(s0, 1e-12))
    u0 = np.log(np.maximum(w0, 1e-12))
    x0 = np.concatenate([v0, u0])
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'xtol': 1e-10, 'disp': False})
        
        if res.x is not None and np.all(np.isfinite(res.x)):
            v = np.clip(res.x[:n], -20, 20)
            u = np.clip(res.x[n:], -20, 20)
            s = np.exp(v)
            w_unnorm = np.exp(u - np.max(u))
            w_sum = np.sum(w_unnorm)
            if w_sum > 0 and np.isfinite(w_sum):
                w = w_unnorm / w_sum
                p = _compute_P_fast(s, w)
                if np.isfinite(p):
                    return s, w, p
    except Exception:
        pass
    
    return None, None, -1.0

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_p = -1.0
    best_s = None
    best_w = None
    
    def consider(s, w):
        nonlocal best_p, best_s, best_w
        if s.size < 1:
            return
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        if np.any(s < -1e-12) or np.any(w < -1e-12):
            return
        if w.sum() <= 1e-12:
            return
        w = w / w.sum()
        p = _compute_P_fast(s, w)
        if p > best_p:
            best_p = p
            best_s = s.copy()
            best_w = w.copy()
    
    # --- Analytical 2-point structure ---
    # For a < b/2: P = (1-p)(3p^2 - 2p^3)
    # dP/dp = 6p - 15p^2 + 8p^3 = p(6 - 15p + 8p^2)
    # 8p^2 - 15p + 6 = 0 -> p = (15 ± sqrt(33))/16
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    P_2pt = (1-p_opt)*(3*p_opt**2 - 2*p_opt**3)
    
    # Use multiple scale factors to explore the (a,b) space
    for a in [1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5]:
        for b in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            if a < b/2:
                s_test = np.array([a, b])
                w_test = np.array([p_opt, 1-p_opt])
                consider(s_test, w_test)
    
    # Also try the other regime: a >= b/2, P = (1-p)*p^3, p* = 3/4
    p_opt2 = 0.75
    for a in [0.6, 0.7, 0.8, 1.0]:
        for b in [1.0, 2.0, 5.0]:
            if a >= b/2:
                s_test = np.array([a, b])
                w_test = np.array([p_opt2, 1-p_opt2])
                consider(s_test, w_test)
    
    # --- Optimize 3-point and 4-point structures ---
    # Strategy: fix the "structure" (which points are small, which large) and optimize
    
    for n in [3, 4, 5]:
        if time.time() > deadline - 60:
            break
        
        # Multiple restarts with different initial structures
        for restart in range(8):
            if time.time() > deadline - 60:
                break
            
            try:
                # Generate diverse initial conditions
                if restart == 0:
                    # Base structure: geometric progression
                    target_s = np.array([0.01, 1.0, 10.0, 100.0, 1000.0])[:n]
                    target_w = np.array([0.5, 0.3, 0.15, 0.05, 0.05])[:n]
                elif restart == 1:
                    # Heavy tail
                    target_s = np.array([0.001, 0.1, 1.0, 10.0, 100.0])[:n]
                    target_w = np.array([0.6, 0.25, 0.1, 0.04, 0.01])[:n]
                elif restart == 2:
                    # Spread out
                    target_s = np.array([0.1, 2.0, 5.0, 20.0, 50.0])[:n]
                    target_w = np.array([0.4, 0.3, 0.2, 0.08, 0.02])[:n]
                elif restart == 3:
                    # Clustered
                    target_s = np.array([0.01, 0.1, 1.0, 2.0, 5.0])[:n]
                    target_w = np.array([0.5, 0.25, 0.15, 0.08, 0.02])[:n]
                else:
                    # Random perturbation of best found so far or random
                    if best_s is not None and best_s.size == n:
                        target_s = best_s * np.exp(np.random.normal(0, 0.3, n))
                        target_w = best_w * np.exp(np.random.normal(0, 0.3, n))
                    else:
                        target_s = np.exp(np.random.uniform(-2, 3, n))
                        target_w = np.exp(np.random.uniform(-2, 0, n))
                
                target_s = np.maximum(target_s, 1e-6)
                target_w = np.maximum(target_w, 1e-6)
                target_w = target_w / target_w.sum()
                
                # L-BFGS-B optimization
                s, w, p = _optimize_n(n, target_s, target_w, maxiter=2000, deadline=deadline)
                if s is not None and p > best_p:
                    best_p = p
                    best_s = s.copy()
                    best_w = w.copy()
                
                # Nelder-Mead polish if time permits
                if time.time() < deadline - 30:
                    s2, w2, p2 = _optimize_n_nm(n, s if s is not None else target_s, 
                                                  w if w is not None else target_w, 
                                                  maxiter=1000, deadline=deadline)
                    if s2 is not None and p2 > best_p:
                        best_p = p2
                        best_s = s2.copy()
                        best_w = w2.copy()
                        
            except Exception:
                pass
    
    # Fallback
    if best_s is None:
        best_s = np.array([0.0, 1.0, 3.0])
        best_w = np.array([0.5, 0.3, 0.2])
        best_w = best_w / np.sum(best_w)
    
    # Validate
    if not (np.all(best_s >= 0) and np.all(best_w >= 0) and np.sum(best_w) > 1e-9):
        n = 32
        best_s = np.arange(n, dtype=np.float64)
        best_w = np.ones(n, dtype=np.float64) / n
    
    return best_s, best_w
