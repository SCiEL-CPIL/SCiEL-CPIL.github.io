"""Rewritten program: multi-start L-BFGS-B on chirp + sinusoid parameterization with smooth-max surrogate."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _value(h):
    """True objective."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return float('inf')
    s = np.sum(h)
    if abs(s) < 1e-10:
        return float('inf')
    c = np.convolve(h, h)
    max_c = np.max(c)
    return abs(2.0 * N * max_c / (s * s))

def _smooth_max_value(h, tau=50.0):
    """Differentiable surrogate for the objective using smooth max."""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = np.convolve(h, h)
    # Smooth max: (1/tau) * log(sum(exp(tau * c / s^2))) * s^2 ... 
    # Actually: max(c) ≈ (1/tau) * log(sum(exp(tau * c)))
    # But values can be large; normalize
    c_norm = c / (s * s)
    max_c_norm = (1.0 / tau) * np.log(np.sum(np.exp(tau * c_norm)))
    return abs(2.0 * N * max_c_norm)

def _build_h(params):
    """Build h from parameters: B, A, f, k, phi, A2, f2, phi2"""
    A, B, f, k, phi, A2, f2, phi2 = params
    i = np.arange(N)
    t = i / N
    h = B + A * np.cos(2.0 * np.pi * f * t + np.pi * k * t * t + phi)
    h += A2 * np.cos(2.0 * np.pi * f2 * t + phi2)
    return h

def _objective(params, tau=50.0):
    h = _build_h(params)
    return _smooth_max_value(h, tau)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_value = 2.0
    
    bounds = [
        (-3.0, 3.0),    # A
        (0.05, 3.0),    # B
        (0.5, 20.0),    # f (chirp frequency)
        (-20.0, 20.0),  # k (chirp rate)
        (0.0, 2 * np.pi),  # phi
        (-2.0, 2.0),    # A2
        (0.5, 20.0),    # f2
        (0.0, 2 * np.pi)   # phi2
    ]
    
    # Multiple diverse starting points
    starts = []
    # Start 1: pure chirp, small B
    starts.append(np.array([1.0, 0.1, 5.0, 10.0, 0.0, 0.0, 1.0, 0.0]))
    # Start 2: different chirp
    starts.append(np.array([1.5, 0.2, 8.0, -15.0, 1.0, 0.5, 3.0, 0.5]))
    # Start 3: low frequency
    starts.append(np.array([0.5, 0.5, 2.0, 5.0, 0.5, 0.3, 5.0, 1.0]))
    # Start 4: high chirp
    starts.append(np.array([2.0, 0.1, 12.0, 18.0, 0.0, 0.0, 2.0, 0.0]))
    # Start 5: negative A
    starts.append(np.array([-1.0, 0.3, 6.0, -8.0, 1.5, -0.5, 4.0, 0.8]))
    # Start 6: moderate
    starts.append(np.array([1.0, 0.15, 7.0, 12.0, 0.3, 0.2, 6.0, 0.2]))
    
    for idx, x0 in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        try:
            # First pass with moderate tau
            res = minimize(
                _objective, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_value:
                h = _build_h(res.x)
                true_val = _value(h)
                if true_val < best_value:
                    best_value = true_val
                    best_h = h
            
            # Second pass with higher tau for sharper max approximation
            if time.time() > deadline - 15.0:
                break
            res2 = minimize(
                lambda p: _smooth_max_value(_build_h(p), tau=200.0), 
                res.x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.fun < best_value:
                h = _build_h(res2.x)
                true_val = _value(h)
                if true_val < best_value:
                    best_value = true_val
                    best_h = h
        except Exception:
            pass
    
    # Final local refinement on best_h using direct perturbation
    if time.time() > deadline - 5.0:
        return best_h
    
    # Try a few random perturbations of best_h
    rng = np.random.RandomState(42)
    for _ in range(20):
        if time.time() > deadline - 3.0:
            break
        h_pert = best_h + rng.randn(N) * 0.01 * np.max(np.abs(best_h))
        val = _value(h_pert)
        if val < best_value:
            best_value = val
            best_h = h_pert
    
    return best_h
