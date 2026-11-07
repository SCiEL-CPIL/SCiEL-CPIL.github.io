"""Bootstrap portfolio: model-and-optimize for Erdos minimum overlap."""
import numpy as np
import time

N = 100
DEADLINE = 140.0  # safety margin below 150s kill
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    # h: (N,) in [0,1], sum ~ N/2
    # value = max(correlate(h, 1-h, full)) / N * 2
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_smooth(h, beta=100.0):
    """Smooth surrogate using log-sum-exp approximation of max."""
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    c_shifted = c - c_max
    s = np.sum(np.exp(beta * c_shifted))
    return (c_max + (1.0/beta) * np.log(s)) / N * 2.0

def project(h):
    """Project to [0,1]^N and fix sum to N/2 (within tolerance)."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    target = N / 2.0
    if s > 0:
        # scale, then re-clip; do a few iterations
        for _ in range(100):
            h = np.clip(h, 0.0, 1.0)
            s = h.sum()
            if abs(s - target) < 1e-12:
                break
            h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h

# Divisors of 100 for spectral basis
DIVISORS_100 = [1, 2, 4, 5, 10, 20, 25, 50]

def cosine_basis_h(coefs):
    """
    h[i] = 0.5 + coefs[0] + sum_k coefs[1+k] * cos(2*pi*f_k*i/N)
    where f_k are divisors of 100 (excluding 0).
    This captures anti-periodic and cyclic structures.
    """
    i = np.arange(N)
    h = np.full(N, 0.5 + coefs[0])
    for k, f in enumerate(DIVISORS_100):
        h = h + coefs[1+k] * np.cos(2*np.pi*f*i/N)
    return h

def optimize_cosine_basis(coefs0, max_iter=3000, beta=50.0):
    """
    Optimize cosine coefficients using SLSQP with epigraph formulation.
    We minimize t subject to:
      max(correlate(h, 1-h)) <= t*N/2
      sum(h) = N/2
      0 <= h <= 1
    """
    from scipy.optimize import minimize
    
    n_coefs = len(coefs0)
    
    def obj_and_grad(x):
        """
        x = [coefs..., t]
        Minimize t (scaled) using smooth surrogate.
        """
        coefs = x[:n_coefs]
        t = x[n_coefs]
        
        h = project(cosine_basis_h(coefs))
        c = np.correlate(h, 1.0 - h, mode="full")
        c_max = np.max(c)
        
        # Smooth max approximation
        c_shifted = c - c_max
        s = np.sum(np.exp(beta * c_shifted))
        soft_max = c_max + (1.0/beta) * np.log(s)
        
        # We want soft_max <= t*N/2
        # Penalty if violated
        violation = max(0, soft_max - t * N / 2.0)
        
        # Objective: t + penalty
        val = t + 100.0 * violation
        return val
    
    # Initial t estimate
    h0 = project(cosine_basis_h(coefs0))
    c0 = np.correlate(h0, 1.0 - h0, mode="full")
    t0 = np.max(c0) / (N / 2.0) + 0.1
    
    x0 = np.concatenate([coefs0, [t0]])
    
    # Bounds for coefficients: keep h in [0,1] roughly
    # Each cos term has amplitude 1, so coefficient should be bounded
    coefs_bounds = [(-0.5, 0.5)] * n_coefs
    t_bounds = [(0.0, 2.0)]
    bounds = coefs_bounds + t_bounds
    
    try:
        res = minimize(obj_and_grad, x0, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-10, "maxls": 100})
        return res.x[:n_coefs]
    except Exception:
        return coefs0

def high_precision_polish(h, max_iter=5000):
    """
    Final polish: use direct coordinate descent with very small steps
    to fine-tune each h[i] while maintaining constraints.
    """
    best_h = h.copy()
    best_val = objective(h)
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if remaining() < 3.0:
            break
            
        # Pick a random index
        idx = rng.integers(0, N)
        
        # Try small perturbations at this index
        for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
            h_new = best_h.copy()
            h_new[idx] += delta
            h_new = project(h_new)
            v = objective(h_new)
            if v < best_val - 1e-15:
                best_h = h_new
                best_val = v
                break
    
    return best_h, best_val

def main_search():
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0

    rng = np.random.default_rng(0)
    
    n_div = len(DIVISORS_100)
    n_coefs = 1 + n_div  # dc offset + one coefficient per divisor
    
    # Multi-start optimization over cosine basis
    n_trials = 60
    for trial in range(n_trials):
        if remaining() < 15.0:
            break
            
        # Random initialization
        coefs0 = np.zeros(n_coefs)
        coefs0[0] = rng.uniform(-0.02, 0.02)
        for k in range(n_div):
            coefs0[1+k] = rng.uniform(-0.3, 0.3)
        
        # Optimize
        coefs_opt = optimize_cosine_basis(coefs0, max_iter=2000, beta=50.0)
        h = project(cosine_basis_h(coefs_opt))
        v = objective(h)
        
        if v < best_val:
            best_val = v
            best_h = h.copy()
            
        if remaining() < 25.0:
            break
    
    # High-precision polish
    if remaining() > 10.0:
        best_h, best_val = high_precision_polish(best_h)
    
    # Final direct local search with adaptive steps
    step_size = 0.02
    for i in range(3000):
        if remaining() < 5.0:
            break
        step_size = 0.02 * max(0.1, 1.0 - i/5000.0)
        h = best_h + rng.normal(0, step_size, N)
        h = project(h)
        v = objective(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    return best_h, best_val

# EVOLVE-BLOCK-START
def construct():
    global _CACHED
    try:
        if '_CACHED' in globals():
            return _CACHED
        h, _ = main_search()
        h = project(h)
        # enforce sum within tolerance
        s = h.sum()
        h = h + (N/2.0 - s)/N
        h = np.clip(h, 0.0, 1.0)
        assert np.all(np.isfinite(h))
        _CACHED = h
        return h
    except Exception:
        return np.full(N, 0.5)
# EVOLVE-BLOCK-END
