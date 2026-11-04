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
from scipy.signal import chirp

N = 400

# EVOLVE-BLOCK-START

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

def _generate_signal(params):
    """
    Generate a signal from a set of parameters.
    Parameters:
    - First 3*K params: (amp, freq, phase) for K sinusoids
    - Next 3 params: (amp, f0, f1) for a chirp
    - Last param: DC offset (to control integral)
    
    We use K=4 sinusoids + 1 chirp = 4*3 + 3 + 1 = 16 parameters.
    """
    K = 4
    x = np.zeros(N)
    
    # Sinusoids
    for i in range(K):
        amp = params[i*3]
        freq = params[i*3 + 1]
        phase = params[i*3 + 2]
        # freq is in cycles/sample. We want incommensurate frequencies to avoid periodicity.
        # Map freq parameter [0, 1] to a specific range of frequencies.
        # Use a range that covers the Nyquist but avoids very low frequencies which create large peaks.
        # Actually, for flat spectrum, we want energy spread out.
        # Let's map freq to [0.05, 0.45] cycles/sample (0.1 to 0.9 Nyquist)
        f_real = 0.05 + 0.4 * freq
        t = np.arange(N)
        x += amp * np.cos(2 * np.pi * f_real * t + phase)
        
    # Chirp
    amp_c = params[K*3]
    f0 = params[K*3 + 1]
    f1 = params[K*3 + 2]
    # Map f0, f1 to [0.05, 0.45]
    f0_real = 0.05 + 0.4 * f0
    f1_real = 0.05 + 0.4 * f1
    t = np.arange(N)
    x += amp_c * chirp(t, f0_real, 1.0, f1_real, method='linear')
    
    # DC Offset
    dc = params[K*3 + 3]
    x += dc
    
    return x

def _objective(params):
    """Objective function for the optimizer. Lower is better."""
    x = _generate_signal(params)
    return _metric(x)

def _optimize_params(deadline, rng):
    """
    Run Differential Evolution to find optimal parameters.
    """
    K = 4
    n_params = K * 3 + 3 + 1  # 12 + 3 + 1 = 16
    
    # Bounds for parameters
    # Amps: [0, 1]
    # Freqs: [0, 1] (mapped internally)
    # Phases: [0, 2*pi]
    # DC: [-1, 1]
    
    bounds = []
    for i in range(K):
        bounds.append((0, 1)) # amp
        bounds.append((0, 1)) # freq
        bounds.append((0, 2 * np.pi)) # phase
    bounds.append((0, 1)) # chirp amp
    bounds.append((0, 1)) # chirp f0
    bounds.append((0, 1)) # chirp f1
    bounds.append((-1, 1)) # dc
    
    # Initial population
    pop_size = 50
    
    # We need to be careful with the time limit.
    # DE can be slow. We will run it with a maxiter that fits in the time, 
    # or simply rely on the timeout mechanism if we wrap it.
    # However, DE doesn't have a native "time limit" stop condition in scipy easily 
    # without a callback. 
    # We can use a callback to check time.
    
    start_time = time.time()
    
    def callback(xk, convergence):
        if time.time() > deadline - 5:
            return True  # Stop optimization
        return False

    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=int(rng.integers(0, 2**31)), 
            maxiter=100, 
            popsize=pop_size, 
            tol=1e-4, 
            callback=callback,
            polish=True,
            init='sobol'
        )
        
        if res.success or res.fun < float('inf'):
            return res.x, res.fun
        else:
            # If it failed, maybe return the best found?
            # DE usually returns the best even if not converged.
            return res.x, res.fun
            
    except Exception as e:
        # Fallback to a random guess if DE crashes
        params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        val = _objective(params)
        return params, val

def _polish_local(x, deadline, rng):
    """
    After finding a good parametric signal, do a local search on the 
    actual 400-dimensional signal to fine-tune.
    This exploits the fact that we started from a good global structure.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Simple local search: flip small subsets of signs or adjust values slightly
    # Since the signal is continuous, we can do gradient-free local search.
    # Let's try a simple coordinate descent or random perturbation.
    
    step_size = 0.1
    no_improve = 0
    
    for i in range(5000):
        if time.time() > deadline - 2:
            break
            
        if no_improve > 100:
            step_size *= 0.9
            if step_size < 0.001:
                break
            no_improve = 0
            
        # Pick a few indices to perturb
        k = rng.integers(1, 5)
        indices = rng.choice(N, size=k, replace=False)
        
        x_new = best_x.copy()
        for idx in indices:
            x_new[idx] += rng.normal(0, step_size)
            
        v_new = _metric(x_new)
        if v_new < best_val:
            best_val = v_new
            best_x = x_new
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def construct():
    deadline = time.time() + 130.0  # Safety margin
    
    rng = np.random.default_rng(2024)
    
    # 1. Global Optimization of Parameters
    # We have ~110 seconds. DE with 16 params should be fast enough.
    params, val = _optimize_params(deadline - 20, rng)
    
    # Generate the signal from the best params
    x_best = _generate_signal(params)
    
    # 2. Local Polish
    # Use the remaining time to polish the specific 400-dim instance.
    x_best, val_best = _polish_local(x_best, deadline, rng)
    
    # 3. Compare with a simple baseline (all ones) just in case
    x_ones = np.ones(N)
    val_ones = _metric(x_ones)
    
    if val_ones < val_best:
        x_best = x_ones
        val_best = val_ones
        
    return x_best

# EVOLVE-BLOCK-END
