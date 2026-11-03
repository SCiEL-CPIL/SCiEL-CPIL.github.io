"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy import optimize
from scipy.optimize import linprog

N = 50

def _score(h):
    """Compute the objective: (sum h^2) / (sum h * max autoconv)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return -np.inf
    h = np.maximum(h, 0.0)
    s2 = np.sum(h**2)
    s1 = np.sum(h)
    if s1 < 1e-15 or s2 < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    if max_conv < 1e-15:
        return 0.0
    return s2 / (s1 * max_conv)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _score(best_h)
    
    def _lp_solve():
        """Solve the LP: minimize max autoconv subject to sum h=1, h>=0"""
        # Autoconvolution of h with itself has 2N-1 = 99 entries
        # conv[k] = sum_i h[i]*h[i+k-N+1] for appropriate k
        # Actually, conv[k] = sum_j h[j]*h[k-j] where the sum is over valid j
        # For k in 0..2N-2: conv[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j]*h[k-j]
        # This is quadratic, not linear! The LP approach only works if we fix the
        # structure. Let me reconsider.
        # Actually, if we fix sum h_i = 1, the objective is ||h||_2^2 / max_conv
        # We want to maximize this. Equivalent to minimizing max_conv / ||h||_2^2
        # This is still not linear. The LP formulation in the directive might be
        # for a different approach. Let me just use a careful multi-start approach.
        pass
    
    # 1. Generate diverse initial patterns
    patterns = []
    
    # Single spike
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        patterns.append(h)
        
    # Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            patterns.append(h)
            
    # Random sparse
    np.random.seed(42)
    for _ in range(50):
        h = np.zeros(N)
        k = np.random.randint(1, 15)
        idx = np.random.choice(N, k, replace=False)
        h[idx] = 1.0
        patterns.append(h)
        
    # Random dense
    for _ in range(50):
        h = np.random.exponential(1.0, size=N)
        patterns.append(h)
        
    # Structured
    patterns.append(np.ones(N))
    patterns.append(np.linspace(0, 1, N))
    patterns.append(np.linspace(1, 0, N))
    patterns.append(np.cos(np.pi * np.linspace(-0.5, 0.5, N))**2)
    patterns.append(np.sin(np.pi * np.linspace(0, 1, N))**2)
    
    # Evaluate patterns
    for h in patterns:
        if time.time() > deadline:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # 2. Multi-start local optimization
    def neg_score(h):
        h = np.maximum(h, 0.0)
        s = _score(h)
        return -s if np.isfinite(s) else 1e10
    
    # Gather diverse starting points
    starts = [best_h.copy()]
    for h in patterns[:20]:
        starts.append(h.copy())
    # Add more random starts
    for _ in range(30):
        h = np.random.exponential(1.0, size=N)
        h /= np.sum(h) * 0.5  # scale
        starts.append(h)
    
    for h0 in starts:
        if time.time() > deadline - 5:
            break
        try:
            res = optimize.minimize(
                neg_score, h0, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            h_opt = np.maximum(res.x, 0.0)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # 3. Final high-precision polish
    if time.time() < deadline - 2:
        try:
            res = optimize.minimize(
                neg_score, best_h, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            h_opt = np.maximum(res.x, 0.0)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # 4. Final check
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END
