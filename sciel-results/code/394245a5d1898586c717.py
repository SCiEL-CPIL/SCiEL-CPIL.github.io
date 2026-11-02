import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    def ratio(h):
        """Compute the proxy ratio: (sum h^2) / ((sum h) * max_k conv_k)"""
        h = np.maximum(h, 0)
        s2 = np.sum(h**2)
        s1 = np.sum(h)
        if s1 < 1e-15:
            return 0.0
        # Compute all lags of autoconvolution
        conv = np.correlate(h, h, mode='full')
        # conv has length 2*N-1, lag 0 is at index N-1
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 0.0
        return s2 / (s1 * max_conv)
    
    def neg_ratio(h):
        return -ratio(h)
    
    # Start from known good point
    h0 = np.zeros(N)
    h0[0] = 2.0/3.0
    h0[1] = 1.0/3.0
    
    best_h = h0.copy()
    best_score = ratio(h0)
    
    # Try different initial points and optimize
    initial_points = [
        h0.copy(),
        np.array([1.0] + [0.0]*(N-1)),
        np.array([0.5, 0.5] + [0.0]*(N-2)),
        np.array([0.4, 0.4, 0.2] + [0.0]*(N-3)),
        np.array([0.5, 0.25, 0.25] + [0.0]*(N-3)),
        np.array([0.6, 0.2, 0.2] + [0.0]*(N-3)),
        np.array([0.4, 0.3, 0.3] + [0.0]*(N-3)),
    ]
    
    # Also try uniform-ish distributions
    for k in [2, 3, 4, 5]:
        pt = np.zeros(N)
        pt[:k] = 1.0/k
        initial_points.append(pt)
    
    for h_init in initial_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(neg_ratio, h_init, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            h_opt = np.maximum(res.x, 0)
            sc = ratio(h_opt)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Local polish with coordinate descent
    if time.time() < deadline:
        h = best_h.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                for delta in [0.01, 0.001, 0.0001, -0.01, -0.001, -0.0001, 0.05, -0.05]:
                    h_try = h.copy()
                    h_try[i] = max(0, h_try[i] + delta)
                    sc = ratio(h_try)
                    if sc > ratio(h) + 1e-12:
                        h = h_try.copy()
                        improved = True
                        break
            # Normalize? No, the ratio is scale-invariant in a specific way
            # Actually (sum h^2)/((sum h)*max conv) is NOT scale invariant
            # If we scale h by c: (c^2 sum h^2)/((c sum h)*(c max conv)) = same
            # So it IS scale invariant! Good.
    
    # Final check
    final_score = ratio(best_h)
    if final_score < best_score:
        # Use the better one
        pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
