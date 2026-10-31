import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
DEADLINE = 140  # seconds, safety margin before 150s kill

def _autoconv_max(h):
    """Compute max of discrete autoconvolution for step heights."""
    # Discrete autoconvolution: (h*h)[k] = sum_i h[i]*h[i+k]
    n = len(h)
    max_val = 0.0
    for k in range(n):
        s = 0.0
        for i in range(n - k):
            s += h[i] * h[i + k]
        if s > max_val:
            max_val = s
    return max_val

def _score(h):
    """Compute the ratio: sum(h^2) / (sum(h) * max_autoconv)."""
    h = np.asarray(h, dtype=float)
    s2 = np.sum(h ** 2)
    s1 = np.sum(h)
    if s1 < 1e-15:
        return 0.0
    amax = _autoconv_max(h)
    if amax < 1e-15:
        return 0.0
    return s2 / (s1 * amax)

def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Deadline guard
    def elapsed():
        return time.time() - start_time
    
    # Try various structured patterns
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Triangular symmetric
    tri = np.linspace(0, 1, N)
    tri = np.concatenate([tri, tri[N-2:0:-1]])
    candidates.append(tri)
    
    # Asymmetric ramp
    ramp = np.linspace(0, 1, N)
    candidates.append(ramp)
    
    # Concentrated center
    for center in [N//4, N//2, 3*N//4]:
        h = np.zeros(N)
        h[center] = 1.0
        candidates.append(h)
    
    # Two adjacent
    for i in range(0, N-1, N//10):
        h = np.zeros(N)
        h[i] = 1.0
        h[i+1] = 1.0
        candidates.append(h)
    
    # Evaluate candidates
    for h in candidates:
        if elapsed() > DEADLINE:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization using Nelder-Mead or similar
    if elapsed() < DEADLINE - 5:
        # Use a smooth surrogate: minimize -log(score) or maximize score directly
        # Since score = s2/(s1*amax), we can try to maximize it
        # Use differential evolution for global search
        try:
            bounds = [(0, 1)] * N
            def neg_score(h):
                return -_score(h)
            
            # Quick DE with limited iterations
            de_result = differential_evolution(
                neg_score, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-4, polish=False, init='latinhypercube'
            )
            if elapsed() < DEADLINE - 3:
                h_de = de_result.x
                s_de = _score(h_de)
                if s_de > best_score:
                    best_score = s_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Polish with local optimization
    if elapsed() < DEADLINE - 2:
        try:
            def neg_score(h):
                return -_score(h)
            
            # Nelder-Mead polish
            res = minimize(
                neg_score, best_h, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if elapsed() < DEADLINE - 1:
                h_pol = res.x
                # Project to nonnegative
                h_pol = np.maximum(h_pol, 0)
                s_pol = _score(h_pol)
                if s_pol > best_score:
                    best_score = s_pol
                    best_h = h_pol.copy()
        except Exception:
            pass
    
    # Final deadline check
    if elapsed() > DEADLINE:
        return list(best_h)
    
    return list(best_h)
