"""
Strategy: Direct coordinate-wise coordinate descent with smooth surrogate objective.

Instead of restricting to a Fourier subspace, we work directly on the full 400-dim
vector h. The key insight is that the objective V = |2*N*max(conv(h,h))/sum(h)^2|
is scale-invariant, so we can fix sum(h)=N (i.e., mean=1) and minimize max(conv(h,h)).

We use a smooth surrogate for the max: max(x) ≈ logsumexp(x/τ)/τ for small τ,
which is differentiable and allows gradient-based optimization. We then do
coordinate-wise updates: for each index i, adjust h[i] to reduce the surrogate.

This is structurally different from the Fourier approach because:
1. It operates in the full 400-dim space, not a 31-dim subspace
2. It uses a smooth surrogate for the non-smooth max
3. It uses coordinate descent rather than L-BFGS on a reduced parameterization
"""

import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value exactly."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _surrogate_max(conv, tau=0.5):
    """Smooth approximation of max(conv) using logsumexp."""
    # Numerically stable logsumexp
    m = np.max(conv)
    if m > 1e10:
        return m
    return m + tau * np.log(np.mean(np.exp((conv - m) / tau)))

def _surrogate_objective(h, tau=0.5):
    """Smooth surrogate objective: surrogate_max(conv(h,h)) / N, with sum(h) fixed to N."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    # Scale invariance: normalize so sum(h) = N
    h_norm = h * N / s
    conv = np.convolve(h_norm, h_norm)
    return _surrogate_max(conv, tau) / N

def _coordinate_update(h, i, tau=0.5, max_iter=50):
    """
    Find optimal value for h[i] given all other coordinates fixed.
    We minimize the surrogate over h[i].
    """
    h_fixed = h.copy()
    h_old = h[i]
    
    def obj(delta):
        h_new = h_fixed.copy()
        h_new[i] = h_old + delta
        return _surrogate_objective(h_new, tau)
    
    try:
        res = minimize_scalar(obj, bounds=(-2.0, 2.0), method='bounded',
                              options={'maxiter': max_iter})
        if res.success:
            return h_old + res.x
    except Exception:
        pass
    return h_old

def construct():
    """
    Construct step heights using coordinate descent with smooth surrogate.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Start from ones
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(123)
    
    # Multiple rounds of coordinate descent with decreasing tau
    taus = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    
    for tau in taus:
        if time.time() > deadline - 20:
            break
        
        h_curr = best_h.copy()
        
        # Normalize to sum = N
        s = np.sum(h_curr)
        if abs(s) > 1e-6:
            h_curr = h_curr * N / s
        
        # Several passes of coordinate descent
        n_passes = 5
        for p in range(n_passes):
            if time.time() > deadline - 15:
                break
            
            # Randomize order to avoid systematic bias
            order = rng.permutation(N)
            
            for idx in order:
                if time.time() > deadline - 15:
                    break
                
                new_val = _coordinate_update(h_curr, idx, tau=tau, max_iter=30)
                h_curr[idx] = new_val
            
            # Check improvement
            v_curr = _evaluate(h_curr)
            if v_curr < best_val:
                best_val = v_curr
                best_h = h_curr.copy()
        
        # After each tau round, try a full L-BFGS optimization on the normalized h
        if time.time() < deadline - 10:
            try:
                # Use the current h as starting point for L-BFGS
                def _lbfgs_obj(x):
                    return _surrogate_objective(x, tau=tau)
                
                res = minimize(
                    _lbfgs_obj,
                    h_curr,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                if res.fun < 1e9:
                    h_opt = res.x
                    v_opt = _evaluate(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final fine-tuning: small random perturbations + coordinate updates
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        v_curr = best_val
        tau_final = 0.01
        
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            
            # Random small perturbation
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, 0.1)
            
            # Coordinate update at that index
            new_val = _coordinate_update(h_new, idx, tau=tau_final, max_iter=20)
            h_new[idx] = new_val
            
            v_new = _evaluate(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
        
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr
    
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
