"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Structured Parametric Manifold Optimization.
Instead of optimizing the full N-dimensional hypercube or using random restarts, 
this strategy models the step height sequence h as a structured function of a small 
number of parameters (frequency, amplitude, phase). This reduces the search space 
from 100 dimensions to 2-3 dimensions, allowing for a global optimization (Differential Evolution) 
to efficiently find a structure that naturally has low autocorrelation (like a pure tone with a specific profile).
The pure tone model h_i = A * sin(2*pi*f*i + phi)^2 + offset is chosen because squared sine waves 
have known autocorrelation properties that can be tuned to minimize the maximum overlap.
A local refinement step (SLSQP) is then applied to the full N-dimensional space initialized by the 
best parametric solution to capture any fine-grained improvements.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    Vectorized using FFT for speed.
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

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    # Vectorized gradient computation
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

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
            return True
        return False

    # Strategy: Parametric Manifold Optimization
    # We model h as a function of a few parameters: frequency f, amplitude A, phase phi.
    # h_i = A * sin(2*pi*f*i + phi)^2 + offset
    # We optimize these parameters using Differential Evolution, then refine locally.
    
    # Define the parametric model
    def parametric_h(params):
        f, A, phi = params
        i = np.arange(N)
        # Squared sine wave
        base = A * np.sin(2 * np.pi * f * i + phi) ** 2
        # Center and scale to roughly fit [0,1] and sum N/2
        # The mean of sin^2 is 0.5, so base has mean A/2
        # We want mean to be 0.5, so offset should be 0.5 - A/2
        offset = 0.5 - A / 2.0
        h = base + offset
        return h
    
    # Objective for DE: minimize the max overlap of the parametric h
    def de_objective(params):
        h = parametric_h(params)
        h = _fix_sum(h)
        return _score(h)
    
    # Bounds for DE: f in [0.1, 5.0], A in [0.1, 1.0], phi in [0, 2*pi]
    bounds = [(0.1, 5.0), (0.1, 1.0), (0.0, 2 * np.pi)]
    
    # Run Differential Evolution
    try:
        result = differential_evolution(de_objective, bounds, maxiter=50, popsize=15, 
                                        seed=42, tol=1e-6, polish=False)
        if time.time() - t0 > _DEADLINE - 20:
            # If we're running out of time, just use the best found so far
            pass
        else:
            best_params = result.x
            best_h_param = parametric_h(best_params)
            _update_best(best_h_param)
    except Exception:
        pass
    
    # Local refinement using SLSQP on the full N-dimensional space
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h0 = best_h.copy()
        
        def _run_optimization(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            def obj_func(x):
                val, _ = _smooth_max_grad(x, beta)
                return val
                
            def jac_func(x):
                _, grad = _smooth_max_grad(x, beta)
                return grad
                
            try:
                # SLSQP is robust for this size
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        # Run optimization with moderate beta
        _run_optimization(h0, beta=50.0, maxiter=200, ftol=1e-9)
        
        # If time permits, run a second pass with higher beta for sharper max
        if time.time() - t0 < _DEADLINE - 5:
             _run_optimization(best_h, beta=100.0, maxiter=100, ftol=1e-10)
    
    if best_h is None:
        # Fallback: Random
        best_h = _fix_sum(np.random.uniform(0, 1, N))
    
    best_h = _fix_sum(best_h)
    return best_h
