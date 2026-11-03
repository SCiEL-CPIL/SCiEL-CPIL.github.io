import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate a good Sidon set
    S = [0]
    used_sums = {0}
    for x in range(1, N):
        new_sums = [x + s for s in S]
        new_sums.append(2 * x)
        collision = any(s in used_sums for s in new_sums)
        if not collision:
            S.append(x)
            for s in new_sums:
                used_sums.add(s)
        if time.time() > deadline:
            break
            
    k = len(S)
    indices = np.array(S)
    
    # 2. Define the objective and constraints for SLSQP
    # We minimize M subject to c_k <= M, sum(h) = 1, h >= 0
    # Variables: h (length k), M (scalar)
    # Total variables: k + 1
    
    def autoconv_peak(h):
        # h is vector of length k
        # Compute linear convolution of h with h
        # c = np.convolve(h, h)
        # Return max(c)
        c = np.convolve(h, h)
        return np.max(c)
    
    def objective(v):
        h = v[:k]
        M = v[k]
        # We want to minimize M
        # But we must ensure c_k <= M is satisfied (handled by constraints)
        # If we just minimize M, SLSQP will push M down until it hits the peak.
        # So objective is just M.
        return M
    
    def constraints(v):
        h = v[:k]
        M = v[k]
        c = np.convolve(h, h)
        # Constraints: c_k - M <= 0 for all k
        # This is 99 constraints.
        # SLSQP can handle this.
        cons = []
        for i in range(len(c)):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[k] - np.convolve(v[:k], v[:k])[i]})
        # Sum h = 1
        cons.append({'type': 'eq', 'fun': lambda v: np.sum(v[:k]) - 1})
        return cons
    
    # Initial guess
    h0 = np.ones(k) / k
    M0 = autoconv_peak(h0)
    v0 = np.append(h0, M0)
    
    # Bounds
    bounds = [(0, 1)] * k + [(0, 1)] # M is bounded by 1 (since sum h=1, max c <= 1)
    
    best_h = None
    best_ratio = 0
    
    # Try a few random starts as well
    starts = [v0]
    for _ in range(10):
        if time.time() > deadline:
            break
        h_rand = np.random.dirichlet(np.ones(k))
        M_rand = autoconv_peak(h_rand)
        starts.append(np.append(h_rand, M_rand))
        
    for v_start in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, v_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
            if res.success:
                h_opt = res.x[:k]
                M_opt = res.x[k]
                # Check validity
                if np.min(h_opt) < -1e-8:
                    continue
                h_opt = np.clip(h_opt, 0, None)
                h_opt = h_opt / np.sum(h_opt)
                
                # Calculate ratio
                c_opt = np.convolve(h_opt, h_opt)
                M_val = np.max(c_opt)
                if M_val > 0:
                    ratio = np.sum(h_opt**2) / M_val
                else:
                    ratio = 0
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h_opt
        except Exception:
            pass
            
    # Construct final vector
    h_final = np.zeros(N)
    if best_h is not None:
        h_final[indices] = best_h
    else:
        # Fallback to equal weights
        h_final[indices] = 1.0 / k
        
    return list(h_final)
