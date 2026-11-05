import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400
DEADLINE = 140  # Safety margin for 150s limit

def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return 1e12
    return val

def _make_h_from_params(params, n_harmonics):
    """
    Generate h from sum of cosines.
    params[0] = DC offset
    params[1] = global amplitude scale
    params[2:] = for each harmonic: (freq, phase, amp)
    """
    offset = params[0]
    A = params[1]
    h = np.full(N, offset, dtype=float)
    for k in range(n_harmonics):
        idx = 2 + 3*k
        freq = params[idx]
        phase = params[idx+1]
        amp = params[idx+2]
        h += A * amp * np.cos(2*np.pi*freq*np.arange(N)/N + phase)
    return h

def _obj_params(params, n_harmonics):
    """Objective function for differential evolution."""
    h = _make_h_from_params(params, n_harmonics)
    if not np.all(np.isfinite(h)):
        return 1e12
    return _value(h)

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    # Evaluate seed
    best_value = _value(best_h)
    
    def _check_deadline():
        return time.time() - start_time > DEADLINE
    
    # Strategy: Use sum-of-cosines parameterization
    # This gives us signals with controlled spectral content
    # Optimize frequencies, phases, and amplitudes to minimize max autocorrelation peak
    
    # Try different numbers of harmonic components
    for n_harmonics in [3, 5, 8]:
        if _check_deadline():
            break
        
        # Parameters: [offset, A, freq_0, phase_0, amp_0, freq_1, phase_1, amp_1, ...]
        bounds = [(-5.0, 5.0), (0.5, 3.0)]
        for _ in range(n_harmonics):
            bounds.append((0.01, 0.99))  # frequency
            bounds.append((0.0, 2*np.pi))  # phase
            bounds.append((0.1, 1.0))  # relative amplitude
        
        # Use differential evolution with time-limited iterations
        try:
            result = differential_evolution(
                _obj_params,
                bounds=bounds,
                args=(n_harmonics,),
                maxiter=80,
                popsize=12,
                tol=1e-3,
                seed=42,
                polish=True,
                update_interval=1
            )
        except Exception:
            pass
        
        if _check_deadline():
            break
        
        # Evaluate the best found
        if result.fun < best_value:
            h_candidate = _make_h_from_params(result.x, n_harmonics)
            if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-12:
                val_candidate = _value(h_candidate)
                if val_candidate < best_value:
                    best_value = val_candidate
                    best_h = h_candidate.copy()
    
    # Random search with structured signals as backup
    if not _check_deadline():
        rng = np.random.RandomState(42)
        for trial in range(1000):
            if _check_deadline():
                break
            
            n_harmonics = rng.randint(3, 15)
            offset = rng.uniform(-3, 3)
            A = rng.uniform(0.5, 2.0)
            h = np.full(N, offset, dtype=float)
            for k in range(n_harmonics):
                freq = rng.uniform(0.01, 0.99)
                phase = rng.uniform(0, 2*np.pi)
                amp = rng.uniform(0.1, 1.0)
                h += A * amp * np.cos(2*np.pi*freq*np.arange(N)/N + phase)
            
            if np.all(np.isfinite(h)) and abs(np.sum(h)) > 1e-12:
                val = _value(h)
                if val < best_value:
                    best_value = val
                    best_h = h.copy()
    
    # Final validation
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    else:
        final_val = _value(best_h)
        if final_val < best_value:
            best_value = final_val
    
    return best_h
