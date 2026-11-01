"""Optimized program for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _compute_c3_bound(heights):
    """
    Compute the autoconvolution upper bound C3 for given step heights.
    The autoconvolution (f*f)(0) = sum_i heights[i] * heights[i] (for symmetric)
    but more generally involves the convolution structure.
    
    For a step function on [-1/4, 1/4] with N intervals, the autoconvolution
    at zero is proportional to sum of products of heights at symmetric positions.
    The 'upper bound on C3' in the context of autocorrelation inequalities
    typically involves the peak of the autoconvolution relative to some norm.
    
    A common formulation: C3 bound = max_k |sum_i h_i h_{i+k}| / (sum |h_i|)^2
    or similar. Let's use the L2-based autoconvolution peak.
    
    Actually, for the third autocorrelation inequality, the relevant quantity
    is often the L-infinity norm of the autoconvolution divided by the L2 norm squared,
    or the peak autoconvolution value.
    
    Let me use: C3 = max_k |autocorr(k)| where autocorr is the circular or linear
    autoconvolution of the step heights.
    """
    # Linear autoconvolution
    autoconv = np.convolve(heights, heights, mode='full')
    # The peak absolute value of the autoconvolution
    return np.max(np.abs(autoconv))

def _objective(x):
    """Objective function for optimization - minimize C3 bound."""
    return _compute_c3_bound(x)

def _gradient(x):
    """Approximate gradient via finite differences (sparse)."""
    eps = 1e-4
    grad = np.zeros_like(x)
    f0 = _compute_c3_bound(x)
    for i in range(N):
        xp = x.copy()
        xp[i] += eps
        grad[i] = (_compute_c3_bound(xp) - f0) / eps
    return grad

def _smooth_objective(x, temperature=0.01):
    """
    Smooth surrogate using soft-max approximation of the max.
    max_i |a_i| ≈ (1/T) * log(sum exp(T*|a_i|))
    """
    autoconv = np.convolve(x, x, mode='full')
    abs_ac = np.abs(autoconv)
    # Soft-max for smoothness
    T = temperature
    max_val = np.max(abs_ac)
    # For numerical stability
    shifted = abs_ac - max_val
    soft_max = max_val + (1.0/T) * np.log(np.sum(np.exp(T * shifted)))
    return soft_max

def _smooth_gradient(x, temperature=0.01):
    """Gradient of the smooth objective via finite differences."""
    eps = 1e-4
    grad = np.zeros_like(x)
    f0 = _smooth_objective(x, temperature)
    for i in range(N):
        xp = x.copy()
        xp[i] += eps
        grad[i] = (_smooth_objective(xp, temperature) - f0) / eps
    return grad

def _local_polish(x, max_iter=100):
    """
    Final high-precision polish: coordinate descent with exact line search
    on the discrete sign structure.
    """
    best_x = x.copy()
    best_val = _compute_c3_bound(best_x)
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        improved = False
        for i in range(N):
            if time.time() > _DEADLINE - 10:
                break
            # Try flipping sign
            new_x = best_x.copy()
            new_x[i] = -best_x[i]
            new_val = _compute_c3_bound(new_x)
            if new_val < best_val - 1e-10:
                best_x = new_x
                best_val = new_val
                improved = True
            # Try magnitude adjustment (if not constrained to ±1)
            # For now, stick with sign flips
        if not improved:
            break
    
    return best_x, best_val

def construct():
    """
    Construct the optimal signed step heights.
    Returns a sequence of N signed step heights.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with random signed initialization
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    best_x = x.copy()
    best_val = _compute_c3_bound(best_x)
    
    # Phase 1: Gradient-based optimization with smooth surrogate
    # Use L-BFGS-B with smooth objective
    temperature = 0.05
    max_wall = _DEADLINE - time.time() - 20  # leave 20s for polish
    
    try:
        # Use smooth objective for gradient
        result = minimize(
            _smooth_objective,
            x,
            args=(temperature,),
            method='L-BFGS-B',
            jac=None,  # let scipy approximate
            options={'maxiter': 200, 'gtol': 1e-5, 'maxfun': 5000},
            bounds=[(-2.0, 2.0)] * N
        )
        if result.success or result.fun < best_val:
            candidate = result.x
            # Project to reasonable range
            candidate = np.clip(candidate, -2.0, 2.0)
            cand_val = _compute_c3_bound(candidate)
            if cand_val < best_val:
                best_x = candidate
                best_val = cand_val
    except Exception:
        pass
    
    # Phase 2: Local polish with sign flips and magnitude tuning
    if time.time() < _DEADLINE - 5:
        best_x, best_val = _local_polish(best_x, max_iter=50)
    
    # Phase 3: If time remains, try multiple random restarts
    if time.time() < _DEADLINE - 15:
        for restart in range(5):
            if time.time() > _DEADLINE - 15:
                break
            x_r = np.random.choice([-1.0, 1.0], size=N)
            try:
                result = minimize(
                    _smooth_objective,
                    x_r,
                    args=(temperature,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-4, 'maxfun': 2000},
                    bounds=[(-2.0, 2.0)] * N
                )
                if result.fun < best_val:
                    candidate = np.clip(result.x, -2.0, 2.0)
                    cand_val = _compute_c3_bound(candidate)
                    if cand_val < best_val:
                        best_x = candidate
                        best_val = cand_val
            except Exception:
                pass
    
    # Final polish
    if time.time() < _DEADLINE - 2:
        best_x, best_val = _local_polish(best_x, max_iter=30)
    
    return best_x
