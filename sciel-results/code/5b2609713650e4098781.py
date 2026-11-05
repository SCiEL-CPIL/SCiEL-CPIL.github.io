"""
Strategy: Direct global search with Differential Evolution + Local Polish.

The previous approach used a low-dimensional Fourier parameterization (K=20 modes),
which restricts the solution space to smooth, low-frequency signals. This new strategy
operates directly on the 400-dimensional vector h.

1. Global Phase: Use scipy.optimize.differential_evolution to explore the 400D space.
   DE is robust to non-convexity and can find diverse basins of attraction.
   The objective function uses a log-sum-exp surrogate for the max term to ensure
   differentiability and smoothness for the optimizer's internal steps.
2. Local Phase: Take the best candidate from DE and refine it using L-BFGS-B
   with a very sharp log-sum-exp parameter (high beta) to minimize the actual
   max peak more precisely.

This avoids the bias towards low-frequency solutions inherent in the Fourier approach
and allows for sharper, more complex step patterns that might yield a lower autocorrelation
peak relative to the sum.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400

def _compute_value(h):
    """Compute the exact evaluation value."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def _objective_de(x, beta=10.0):
    """
    Objective for Differential Evolution.
    We want to minimize max(conv) / sum(h)^2.
    Using log-sum-exp with a moderate beta for smoothness.
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    # Smooth max approximation
    m = np.max(conv)
    # logsumexp is numerically stable
    # scipy.special.logsumexp is available, but let's use manual stable version to avoid import overhead if possible
    # Actually, scipy.special is standard. Let's use it.
    from scipy.special import logsumexp
    sm = logsumexp(conv / beta) * beta
    return sm / (s ** 2)

def _objective_local(x, beta=100.0):
    """
    Objective for L-BFGS-B local polish.
    Uses a sharper beta to better approximate the true max.
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    m = np.max(conv)
    from scipy.special import logsumexp
    sm = logsumexp(conv / beta) * beta
    return sm / (s ** 2)

def construct():
    """Construct step heights minimizing autocorrelation peak."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer)
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Initialize DE
    # Bounds: We can scale h. The value is scale-invariant, but numerical stability
    # suggests keeping values in a reasonable range. 
    # Let's fix the scale by normalizing in the objective? 
    # No, DE works better with fixed bounds. 
    # Since the objective is homogeneous of degree 0 (scale invariant), 
    # the optimization landscape has a "valley" along the scaling direction.
    # To help DE, we can constrain the sum to be 1 (or N) by projecting or just letting it wander.
    # However, if sum -> 0, value -> inf. If sum -> inf, value -> 0 (but conv also grows).
    # Wait, value = 2N * max(conv) / sum^2.
    # If h = c * h0, conv = c^2 * conv0, sum = c * sum0.
    # value = 2N * c^2 * max(conv0) / (c^2 * sum0^2) = 2N * max(conv0) / sum0^2.
    # So it is strictly scale invariant.
    # DE might struggle with this flat direction. 
    # A common trick is to normalize h in the objective or constrain the norm.
    # Let's add a soft constraint or just rely on the fact that DE explores.
    # Better: Fix the sum to 1? No, the problem doesn't require it.
    # But we can initialize with sum=N.
    
    # Bounds for DE
    # Let's assume heights are in [-10, 10]. 
    # If the optimal solution has large spikes, this might clip them.
    # But scale invariance means we can always scale down.
    # So restricting to [-10, 10] is safe as long as we don't force sum=0.
    bounds = [(-10.0, 10.0)] * N
    
    # DE Parameters
    # Popsize 15 is standard, but with 400 dims, it's expensive.
    # maxiter: We have ~140s. 
    # Each DE step requires popsize * 2 evaluations (candidate + incumbent) roughly.
    # Let's estimate cost. 
    # 1 eval = 1 convolve (400x400 -> 799) + sum + logsumexp.
    # 400 dims, popsize 15 -> 15*400 = 6000 vars per generation.
    # 20 generations -> 120,000 function calls.
    # If 1 call is ~1ms, that's 120s. Too slow?
    # Let's profile mentally: np.convolve(400, 400) is very fast in C. ~10-50 microseconds.
    # Python overhead is higher. Let's say 100us per call.
    # 120,000 calls * 100us = 12 seconds. This is feasible.
    # We can increase popsize or maxiter.
    
    popsize = 20
    maxiter = 50
    tol = 1e-4
    seed = 42
    
    # Initial bounds might be too tight if the optimal shape requires large dynamic range?
    # No, scale invariance means we can scale any shape to fit [-10, 10].
    
    try:
        result_de = differential_evolution(
            _objective_de,
            bounds,
            args=(10.0,), # beta for DE
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            seed=seed,
            polish=False, # We do our own polish
            init='latinhypercube'
        )
        
        if result_de.success or result_de.fun < best_val:
            h_de = result_de.x
            val_de = _compute_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
                
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline - 20:
        return best_h if np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-10 else np.ones(N)
        
    # Phase 2: Local Polish with L-BFGS-B
    # Use the best solution found so far
    h_start = best_h.copy()
    
    # It might be beneficial to try polishing from a few different points if time permits
    # But DE already did global search. One good polish is usually sufficient.
    
    try:
        result_local = minimize(
            _objective_local,
            h_start,
            method='L-BFGS-B',
            args=(100.0,), # Sharp beta
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        if result_local.fun < best_val:
            h_local = result_local.x
            val_local = _compute_value(h_local)
            if val_local < best_val:
                best_val = val_local
                best_h = h_local.copy()
                
    except Exception:
        pass
        
    # Phase 3: If time remains, try a second local polish from a slightly perturbed version
    # or from the DE initial best if different
    if time.time() < deadline - 5 and best_val < 2.0:
        # Perturb best_h
        h_perturb = best_h + np.random.normal(0, 0.01, N)
        try:
            result_local2 = minimize(
                _objective_local,
                h_perturb,
                method='L-BFGS-B',
                args=(100.0,),
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result_local2.fun < best_val:
                h_local2 = result_local2.x
                val_local2 = _compute_value(h_local2)
                if val_local2 < best_val:
                    best_val = val_local2
                    best_h = h_local2.copy()
        except Exception:
            pass

    # Final validation
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
        
    return best_h
