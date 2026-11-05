"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 400

def _value(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    mx = np.max(conv)
    return abs(2.0 * N * mx / (s * s))

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # 10s safety margin
    
    # Start with a good initial pattern: use a Barker-like or random low-autocorr pattern
    # Then optimize with scipy
    try:
        from scipy.optimize import minimize
    except ImportError:
        return np.ones(N)
    
    # Try multiple initializations
    best_h = np.ones(N, dtype=float)
    best_val = _value(best_h)  # = 2.0
    
    rng = np.random.RandomState(42)
    
    # Strategy: h = a + b * m, where m is a ±1 pattern
    # Optimize over (a, b) for a fixed m, then try several m's
    
    # Generate several low-autocorrelation ±1 patterns
    patterns = []
    
    # Pattern 1: all ones (baseline)
    patterns.append(np.ones(N, dtype=float))
    
    # Pattern 2: alternating
    patterns.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Pattern 3-10: random ±1 patterns, pick ones with low peak autocorr sidelobe
    for trial in range(50):
        m = rng.choice([-1.0, 1.0], size=N)
        conv_m = np.convolve(m, m, mode='full')
        # Remove the center peak
        center = N - 1
        peak = np.max(np.abs(np.delete(conv_m, center)))
        patterns.append((peak, m))
    
    # Sort by peak sidelobe, take best few
    patterns_with_score = [p for p in patterns if isinstance(p, tuple)]
    patterns_with_score.sort(key=lambda x: x[0])
    
    candidate_patterns = [patterns[0], patterns[1]]  # ones, alternating
    for _, m in patterns_with_score[:10]:
        candidate_patterns.append(m)
    
    for m in candidate_patterns:
        if time.time() > deadline - 10:
            break
        
        # Optimize h = a + b*m over (a, b)
        def obj(params):
            a, b = params
            h = a + b * m
            return _value(h)
        
        # Try several initial (a, b)
        inits = [(1.0, 0.0), (1.0, 0.1), (1.0, -0.1), (0.5, 1.0), (1.0, 0.5), (1.0, -0.5)]
        
        for a0, b0 in inits:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(obj, [a0, b0], method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.success or res.fun < best_val:
                    h_candidate = res.x[0] + res.x[1] * m
                    v = _value(h_candidate)
                    if v < best_val and np.all(np.isfinite(h_candidate)) and np.sum(h_candidate) != 0:
                        best_val = v
                        best_h = h_candidate.copy()
            except Exception:
                pass
    
    # If we found something better, try to polish with direct optimization on h
    if best_val < 2.0 and time.time() < deadline - 15:
        try:
            def obj_h(h):
                return _value(h)
            
            # Use L-BFGS-B with bounds to keep values reasonable
            bounds = [(-10, 10)] * N
            
            res = minimize(obj_h, best_h, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                h_candidate = res.x
                v = _value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and np.sum(h_candidate) != 0:
                    best_val = v
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final check
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
