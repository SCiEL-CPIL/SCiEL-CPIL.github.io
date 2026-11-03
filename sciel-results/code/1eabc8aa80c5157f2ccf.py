import numpy as np
import time
from scipy.optimize import minimize, linprog
from itertools import combinations

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 115.0
    
    def elapsed():
        return time.time() - start_time
    
    best_h = None
    best_score = 0.0
    
    def try_update(h):
        nonlocal best_h, best_score
        h = np.maximum(0, np.asarray(h, dtype=np.float64))
        if np.sum(h) <= 1e-15:
            return
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 1: Try structured patterns
    # Pattern 1: k equal heights at specific positions
    for k in range(2, N+1):
        if elapsed() > deadline - 30:
            break
        # Try all subsets of size k is too expensive for large k
        # Instead, try regular spacings
        for spacing in range(1, N):
            if elapsed() > deadline - 30:
                break
            for start in range(min(spacing, N)):
                indices = list(range(start, N, spacing))
                if len(indices) < 2:
                    continue
                h = np.zeros(N)
                h[indices] = 1.0
                try_update(h)
    
    # Phase 2: Epigraph-based optimization
    # Reformulate: maximize L2^2 / (L1 * t) subject to autoconv_k <= t for all k, L1 = 1, h >= 0
    # Equivalent: for fixed t, maximize L2^2 subject to autoconv_k <= t, L1=1, h>=0
    # Or: minimize t - lambda * L2^2 for some lambda
    
    # Better: use the fact that with L1=1, we maximize L2^2 / t
    # Use scipy minimize with epigraph
    
    def neg_obj_epigraph(x):
        """x = [h_0, ..., h_{N-1}, t]"""
        h = x[:N]
        t = x[N]
        if t <= 0:
            return 1e10
        l1 = np.sum(h)
        if l1 <= 1e-15:
            return 1e10
        l2_sq = np.sum(h**2)
        # Penalty for autoconv > t
        h_conv = np.convolve(h, h, mode='full')
        viol = np.maximum(0, h_conv - t)
        penalty = np.sum(viol**2)
        return -(l2_sq / (l1 * t)) + penalty * 1e4
    
    # Start from best found
    if best_h is not None:
        h_start = best_h / np.sum(best_h)
        h_conv_start = np.convolve(h_start, h_start, mode='full')
        t_start = np.max(h_conv_start) * 1.1
        
        x_start = np.concatenate([h_start, [t_start]])
        bounds = [(0, 1)] * N + [(1e-10, None)]
        
        try:
            res = minimize(neg_obj_epigraph, x_start, method='SLSQP', 
                          bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12})
            if res.fun < 1e9:
                h_candidate = res.x[:N]
                try_update(h_candidate)
        except Exception:
            pass
    
    # Phase 3: Coordinate descent with support selection
    # For a given support S, optimize heights on S
    def optimize_on_support(S, h_init=None, max_iter=100):
        """Optimize heights on support S using coordinate descent."""
        if len(S) < 2:
            return None
        
        if h_init is None:
            h = np.zeros(N)
            h[S] = 1.0 / len(S)
        else:
            h = np.asarray(h_init, dtype=np.float64).copy()
            h[~np.isin(np.arange(N), S)] = 0
        
        for it in range(max_iter):
            if elapsed() > deadline - 10:
                break
            improved = False
            for i in S:
                # Optimize h[i] given others fixed
                # L1 = sum_{j!=i} h[j] + h[i] = L1_rest + h[i]
                # L2^2 = sum_{j!=i} h[j]^2 + h[i]^2 = L2_rest + h[i]^2
                # autoconv: each entry is a quadratic in h[i]
                # This is complex; use numerical 1D optimization
                h_old = h[i]
                L1_rest = np.sum(h) - h[i]
                
                def obj_1d(hi):
                    h_test = h.copy()
                    h_test[i] = hi
                    l1 = np.sum(h_test)
                    if l1 <= 1e-15:
                        return 0
                    l2_sq = np.sum(h_test**2)
                    h_conv = np.convolve(h_test, h_test, mode='full')
                    max_c = np.max(h_conv)
                    if max_c <= 1e-15:
                        return 0
                    return l2_sq / (l1 * max_c)
                
                # Use golden section search
                a, b = 0, 1.0
                for _ in range(30):
                    m1 = a + (b - a) * 0.382
                    m2 = a + (b - a) * 0.618
                    if obj_1d(m1) < obj_1d(m2):
                        b = m2
                    else:
                        a = m1
                    if b - a < 1e-10:
                        break
                h_new = (a + b) / 2
                h[i] = h_new
                if abs(h_new - h_old) > 1e-10:
                    improved = True
            
            if not improved:
                break
        
        return h
    
    # Try various support patterns
    support_patterns = []
    
    # Regular grids
    for k in range(2, 15):
        for spacing in range(1, N // max(1, k - 1) + 1):
            for start in range(min(spacing, 5)):
                S = list(range(start, N, spacing))
                if len(S) >= 2 and len(S) <= 20:
                    support_patterns.append(S)
    
    # Random supports
    for _ in range(50):
        if elapsed() > deadline - 20:
            break
        k = np.random.randint(2, 15)
        S = list(np.random.choice(N, k, replace=False))
        support_patterns.append(S)
    
    for S in support_patterns:
        if elapsed() > deadline - 15:
            break
        h_opt = optimize_on_support(S, max_iter=50)
        if h_opt is not None:
            try_update(h_opt)
    
    # Phase 4: Full SLSQP from best
    if best_h is not None and elapsed() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        
        def neg_obj_h(h):
            l1 = np.sum(h)
            if l1 <= 1e-15:
                return 0.0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv <= 1e-15:
                return 0.0
            return -(l2_sq / (l1 * max_conv))
        
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success and res.fun < 0:
                try_update(res.x)
        except Exception:
            pass
    
    if best_h is None:
        best_h = np.ones(N)
    
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)
