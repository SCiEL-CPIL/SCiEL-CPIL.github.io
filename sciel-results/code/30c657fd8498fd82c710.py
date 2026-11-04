"""
Strategy: Low-dimensional parametric optimization (Multi-Sine/Chirp).
Instead of optimizing all 400 values directly, we model the sequence as a sum of
a few sinusoids (chirps) with optimized frequencies and phases. This reduces the
search space from 400 dimensions to ~10-20 dimensions, allowing for much faster
and more thorough exploration using differential evolution or Nelder-Mead.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _generate_signal(params):
    """
    Generate a signal from parameters.
    params: [f1, f2, f3, p1, p2, p3, a1, a2, a3]
    f: frequencies (cycles per sample), p: phases (radians), a: amplitudes
    """
    n = np.arange(N)
    x = np.zeros(N)
    # 3 sinusoidal components
    for i in range(3):
        f = params[i]
        p = params[i + 3]
        a = params[i + 6]
        x += a * np.cos(2 * np.pi * f * n + p)
    return x

def _objective(params):
    x = _generate_signal(params)
    return _metric(x)

def construct():
    deadline = time.time() + 145.0
    
    # Initial best
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Define parameter space
    # Frequencies: 0.0 to 1.0 (Nyquist)
    # Phases: 0 to 2*pi
    # Amplitudes: -1 to 1
    bounds = [
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),  # f1, f2, f3
        (0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi), # p1, p2, p3
        (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)  # a1, a2, a3
    ]
    
    # Phase 1: Differential Evolution (Global Search)
    # Use a moderate population size and maxiter to fit within time budget
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            polish=True,
            init='sobol'
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = _generate_signal(result.x)
    except Exception:
        pass
    
    # Phase 2: Local Refinement (Nelder-Mead or L-BFGS) on the best parametric solution
    if time.time() < deadline - 10:
        try:
            # Start from the best DE result or a random start if DE failed
            if 'result' in locals() and hasattr(result, 'x'):
                x0 = result.x
            else:
                x0 = np.random.uniform(bounds[0][0], bounds[0][1], size=9)
                for i in range(1, 3):
                    x0[i] = np.random.uniform(0, 1)
                for i in range(3, 6):
                    x0[i] = np.random.uniform(0, 2*np.pi)
                for i in range(6, 9):
                    x0[i] = np.random.uniform(-1, 1)
            
            res_local = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_local.fun < best_val:
                best_val = res_local.fun
                best_x = _generate_signal(res_local.x)
        except Exception:
            pass

    # Phase 3: Try a different initial set of parameters (e.g., higher frequencies)
    if time.time() < deadline - 20:
        bounds2 = [
            (0.5, 1.0), (0.3, 0.8), (0.1, 0.5),  # Higher freq ranges
            (0.0, 2*np.pi), (0.0, 2*np.pi), (0.0, 2*np.pi),
            (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)
        ]
        try:
            result2 = differential_evolution(
                _objective,
                bounds2,
                seed=123,
                maxiter=100,
                popsize=10,
                tol=1e-5,
                polish=True
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_x = _generate_signal(result2.x)
        except Exception:
            pass

    # Phase 4: Final local polish on the best found parametric signal
    if time.time() < deadline - 5:
        try:
            # Extract current best parameters? 
            # We don't store them explicitly in best_x, so we need to re-optimize from scratch 
            # or just do a quick local search on the time-domain vector if we had params.
            # Since we only have best_x, we can try a few random perturbations in time domain
            # or just return best_x. 
            # Let's do a quick coordinate descent on the time-domain vector for the best_x found.
            
            # Actually, the parametric signal is a smooth function. 
            # Let's just ensure we return the best parametric result.
            pass
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
