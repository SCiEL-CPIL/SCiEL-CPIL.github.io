"""
Strategy: Low-dimensional parametric optimization (Multi-Sine/Chirp)
combined with Differential Evolution for global search.

Rationale:
1. The objective function (max |autoconv| / integral^2) is highly non-convex in 400D.
2. Local searches (SA, Gradient) often get trapped in local minima.
3. By restricting the search space to a family of signals defined by ~16 parameters 
   (amplitudes, frequencies, phases of a few sinusoids + a chirp component), 
   we can use a global optimizer (Differential Evolution) which is robust to 
   local minima.
4. A "flat spectrum" signal (white noise-like) minimizes the peak-to-average 
   power ratio of the autoconvolution. A sum of incommensurate sinusoids 
   approximates this well.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _parametric_signal(params):
    """
    Generate a signal from low-dimensional parameters.
    params: [A1, f1, p1, A2, f2, p2, A3, f3, p3, A4, f4, p4, A5, f5, p5, chirp_rate]
    Returns a signal of length N.
    """
    t = np.arange(N)
    x = np.zeros(N)
    
    # 5 sinusoidal components
    for i in range(5):
        A = params[i*3]
        f = params[i*3 + 1]
        p = params[i*3 + 2]
        x += A * np.sin(2 * np.pi * f * t / N + p)
        
    # Chirp component (quadratic phase)
    chirp_rate = params[15]
    x += 0.5 * np.sin(2 * np.pi * (0.1 * t / N + 0.5 * chirp_rate * (t / N)**2))
    
    # Normalize to zero mean and unit variance to avoid trivial scaling issues
    # We want the integral to be non-zero, so we add a small DC offset if needed
    # But actually, the metric scales as 1/S^2. If we fix the shape, we can scale x.
    # However, the evaluator likely expects the raw sequence.
    # The metric is homogeneous of degree 0 if we scale x?
    # Let x -> c*x. Autoconv -> c^2 * Autoconv. Integral -> c * Integral.
    # Metric -> c^2 / c^2 = 1. So scaling doesn't change the metric.
    # We just need a valid representative.
    
    return x

def _objective(params):
    """Objective function for the optimizer."""
    x = _parametric_signal(params)
    return _metric(x)

def construct():
    # Deadline guard: return with a safety margin before the 150 s kill.
    deadline = time.time() + 130.0
    
    # Define bounds for the parameters
    # Amplitudes: 0 to 1
    # Frequencies: 1 to 10 (cycles over the signal length)
    # Phases: 0 to 2*pi
    # Chirp rate: -10 to 10
    bounds = []
    for i in range(5):
        bounds.append((0.0, 1.0))  # Amplitude
        bounds.append((1.0, 10.0)) # Frequency
        bounds.append((0.0, 2 * np.pi)) # Phase
    bounds.append((-10.0, 10.0)) # Chirp rate
    
    # Initial best
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Run Differential Evolution
    # maxiter is limited by time
    # popsize=15, tol=0.01
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=0.01, 
            seed=2024,
            polish=True, # Local polish at the end
            init='sobol'
        )
        
        if result.success or result.fun < best_val:
            best_params = result.x
            best_x = _parametric_signal(best_params)
            best_val = _metric(best_x)
            
    except Exception:
        pass
    
    # If we have time left, try a few random initializations with local search
    # Or just return the best parametric solution found.
    
    # Simple local search on the parametric space?
    # DE with polish=True already does Nelder-Mead on the best point.
    
    # Let's do a quick random restart loop for DE if time permits
    if time.time() < deadline - 20:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            try:
                res = differential_evolution(
                    _objective, 
                    bounds, 
                    maxiter=20, 
                    popsize=10, 
                    tol=0.01, 
                    seed=seed,
                    polish=True
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = _parametric_signal(res.x)
            except Exception:
                pass
                
    return best_x
