"""
Strategy: Parametric Chirp/Sinusoid Representation
Instead of optimizing 400 individual heights, we restrict the solution space to a 4-parameter family:
h[i] = A * cos(2*pi*f*i/N + phi) + B
This exploits the structure that low-frequency sinusoids have predictable autocorrelation sidelobes.
We use Differential Evolution to find the global optimum in this 4D space, then Nelder-Mead for local refinement.
This avoids the high-dimensional local minima traps of the full vector optimization.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 145.0  # wall-clock limit with safety margin (150s limit)

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _parametric_h(params):
    """
    Generate h from parameters: [A, f, phi, B]
    h[i] = A * cos(2*pi*f*i/N + phi) + B
    """
    A, f, phi, B = params
    i = np.arange(N)
    return A * np.cos(2 * np.pi * f * i / N + phi) + B

def _objective(params):
    """
    Objective function for the optimizer.
    We minimize the value.
    """
    h = _parametric_h(params)
    val = _evaluate(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    """
    Construct step heights by optimizing a 4-parameter sinusoidal model.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initial best: seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Define bounds for the 4 parameters: [A, f, phi, B]
    # A: amplitude. If A is too large relative to B, sum might go to 0 or negative.
    # We want sum(h) != 0.
    # Let's try a range that allows negative values (important for lowering max autocorr).
    # f: frequency. Low frequencies (1-5 cycles) are most likely to have structured sidelobes.
    # phi: phase.
    # B: offset.
    
    bounds = [
        (-5.0, 5.0),      # A
        (0.5, 10.0),      # f
        (0.0, 2 * np.pi), # phi
        (-5.0, 5.0)       # B
    ]
    
    # Phase 1: Differential Evolution
    # DE is good for global search in low dimensions.
    # maxiter is set conservatively to stay within time limits.
    # tol is tight.
    
    try:
        res = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            seed=42,
            polish=False  # We do our own polish
        )
        
        if time.time() < deadline - 10:
            best_params = res.x
            best_h = _parametric_h(best_params)
            best_val = _evaluate(best_h)
            
            # Phase 2: Local Refinement with Nelder-Mead
            # Start from the DE result
            res2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res2.fun < best_val:
                best_params = res2.x
                best_h = _parametric_h(best_params)
                best_val = res2.fun
                
    except Exception:
        pass

    # Phase 3: Try a few other random initializations in case DE got stuck in a local minimum
    # specifically for the parametric space.
    rng = np.random.default_rng(123)
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            # Random params
            p_rand = np.array([
                rng.uniform(-5, 5),
                rng.uniform(0.5, 10),
                rng.uniform(0, 2*np.pi),
                rng.uniform(-5, 5)
            ])
            
            # Quick local refine
            try:
                res_r = minimize(
                    _objective,
                    p_rand,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_r.fun < best_val:
                    best_params = res_r.x
                    best_h = _parametric_h(best_params)
                    best_val = res_r.fun
            except Exception:
                continue

    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
