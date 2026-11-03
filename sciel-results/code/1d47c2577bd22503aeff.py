"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses SLSQP with adaptive log-sum-exp smoothing (annealing) to navigate the non-convex landscape.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (log-sum-exp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Shift to avoid overflow
    cs_shifted = cs - m
    exp_terms = np.exp(beta * cs_shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # r_k = sum_{i=k}^{n-1} h[i] h[i-k]
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    # Note: The term for j=k is counted once in the sum for r_k?
    # r_k = h_k h_0 + h_{k+1} h_1 + ... + h_{n-1} h_{n-1-k}
    # If we differentiate r_k with respect to h_j:
    # If j is in the first index (i), h_j h_{j-k}. Valid if j>=k.
    # If j is in the second index (i-k), h_{j+k} h_j. Valid if j+k < n.
    # If j=k, the term is h_k h_0. 
    # Wait, the sum is over i from k to n-1.
    # Term i: h_i h_{i-k}.
    # If j = i, term is h_j h_{j-k}. Contribution: h_{j-k}.
    # If j = i-k, term is h_{j+k} h_j. Contribution: h_{j+k}.
    # If j = k, i=k, term h_k h_0.
    # If we vary h_k, it appears as h_k (first factor) in term i=k.
    # It also appears as h_{i-k} in term i=2k? No, i-k=k => i=2k.
    # So if j=k, we have contribution from i=k (h_0) and if 2k < n, from i=2k (h_k? No, h_{2k} h_k).
    # Let's check the formula derived in the previous program:
    # grad[k:] += factor * h[:n-k]  (for j>=k, h_{j-k})
    # grad[:n-k] += factor * h[k:]   (for j+k<n, h_{j+k})
    # This looks correct.
    
    for k in range(n):
        if w[k] < 1e-10:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-8) & (h < 1.0 - 1e-8)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Analytical Start: Cosine Wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    def _run_optimization(h0, beta_start, beta_end, maxiter, ftol):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        # We will do a simple annealing by re-running SLSQP with different betas
        # SLSQP itself doesn't support changing objective parameters easily during run,
        # so we chain them.
        
        current_h = h0.copy()
        
        # Define a helper to run SLSQP with a fixed beta
        def _slsqp_run(h_init, beta):
            def obj_func(x):
                val, _ = _smooth_max_grad(x, beta)
                return val
            def jac_func(x):
                _, grad = _smooth_max_grad(x, beta)
                return grad
            
            try:
                res = minimize(obj_func, h_init, jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < _score(h_init):
                    return res.x
                return h_init
            except Exception:
                return h_init

        # Annealing steps
        betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        # Filter betas within range
        active_betas = [b for b in betas if b_start <= b <= beta_end]
        if not active_betas:
            active_betas = [beta_end]
            
        for b in active_betas:
            if time.time() > _DEADLINE - 5:
                break
            current_h = _slsqp_run(current_h, b)
            _update_best(current_h)
            
        return current_h

    # Phase 1: Refine Cosine Start with Annealing
    if time.time() - t0 < _DEADLINE - 60:
        _run_optimization(h_cos, 10.0, 500.0, 100, 1e-9)
        
    # Phase 2: Random Restarts
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(5):
            if time.time() - t0 > _DEADLINE - 10:
                break
            # Generate a random start that is somewhat smooth to help optimizer
            h_rand = np.random.uniform(0, 1, N)
            # Smooth it slightly
            kernel = np.ones(5) / 5.0
            h_rand = np.convolve(h_rand, kernel, mode='same')
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            
            # Run shorter annealing for random starts
            _run_optimization(h_rand, 20.0, 200.0, 50, 1e-8)
            
    # Phase 3: Final Polish on Best
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        # Very high beta, tight tolerance
        def _final_polish(h_init):
            beta = 1000.0
            def obj_func(x):
                val, _ = _smooth_max_grad(x, beta)
                return val
            def jac_func(x):
                _, grad = _smooth_max_grad(x, beta)
                return grad
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            try:
                res = minimize(obj_func, h_init, jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 200, 'ftol': 1e-12})
                if res.success or res.fun < _score(h_init):
                    return res.x
                return h_init
            except Exception:
                return h_init

        best_h = _final_polish(best_h)
        _update_best(best_h)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
