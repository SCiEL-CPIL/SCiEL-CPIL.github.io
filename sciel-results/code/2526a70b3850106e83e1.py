import time
import math
import numpy as np
from scipy.optimize import minimize, nelder_mead

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h)."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=500.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Search over periods p
    # p must be at least ceil(N/2) to allow sum N/2 with h_i <= 1
    # Actually, sum is N/2, max sum for period p is p. So p >= N/2 = 50.
    for p in range(50, N + 1):
        if time.time() > deadline:
            break
            
        k = N // p
        r = N % p
        
        # We optimize b[0] to b[p-2]. b[p-1] is determined by sum constraint.
        # Sum H = k * sum(b) + sum(b[:r]) = N/2
        # Let b_rest = b[:p-1]
        # b[p-1] is the last element.
        # H = k * (sum(b_rest) + b[p-1]) + sum(b[:r])
        # Note: b[:r] is a subset of b_rest if r <= p-1.
        # If r == 0, b[:r] is empty.
        # If r > 0, b[:r] contains b[0]...b[r-1].
        # Since r < p, r-1 <= p-2, so b[:r] is indeed a subset of b_rest.
        
        # H = k * sum(b_rest) + k * b[p-1] + sum(b_rest[:r])
        # k * b[p-1] = N/2 - k * sum(b_rest) - sum(b_rest[:r])
        # b[p-1] = (N/2 - k * sum(b_rest) - sum(b_rest[:r])) / k
        
        def objective(b_rest):
            # Check bounds for b_rest
            if np.any(b_rest < -1e-9) or np.any(b_rest > 1 + 1e-9):
                return 10.0 # Large penalty
            
            sum_rest = np.sum(b_rest)
            sum_prefix = np.sum(b_rest[:r]) if r > 0 else 0.0
            
            b_last = (N/2.0 - k * sum_rest - sum_prefix) / k
            
            # Check bound for b_last
            if b_last < -1e-9 or b_last > 1 + 1e-9:
                return 10.0 # Large penalty
            
            # Construct full b
            b = np.empty(p)
            b[:p-1] = b_rest
            b[p-1] = b_last
            
            # Construct h
            h = np.tile(b, N // p + 1)[:N]
            
            # Calculate smooth max overlap
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=500.0)
            
            return smax

        # Run Nelder-Mead with multiple random restarts
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initial guess in [0, 1]
            b0 = np.random.rand(p - 1)
            
            try:
                res = nelder_mead(
                    objective, 
                    b0, 
                    xatol=1e-8, 
                    fatol=1e-8, 
                    maxiter=1000, 
                    maxfev=5000
                )
                
                if res.success:
                    b_opt_rest = res.x
                    sum_rest = np.sum(b_opt_rest)
                    sum_prefix = np.sum(b_opt_rest[:r]) if r > 0 else 0.0
                    b_last = (N/2.0 - k * sum_rest - sum_prefix) / k
                    
                    if 0.0 <= b_last <= 1.0:
                        b_opt = np.empty(p)
                        b_opt[:p-1] = b_opt_rest
                        b_opt[p-1] = b_last
                        
                        h_opt = np.tile(b_opt, N // p + 1)[:N]
                        h_opt = _normalize(h_opt)
                        score = _score(h_opt)
                        
                        if score < best_score:
                            best_score = score
                            best_h = h_opt.copy()
                            
            except Exception as e:
                pass

    # Final polish on the best found using full N variables with SLSQP (as before, but maybe shorter)
    if time.time() < deadline - 5.0:
        def obj_full(h):
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=1000.0)
            s = np.sum(h)
            sum_penalty = 1000.0 * (s - N/2.0)**2
            return smax + sum_penalty

        def con_full(h):
            return np.sum(h) - N/2.0

        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(
                obj_full, 
                best_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[{'type': 'eq', 'fun': con_full}],
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                h_opt = _normalize(res.x)
                score = _score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass

    return best_h
