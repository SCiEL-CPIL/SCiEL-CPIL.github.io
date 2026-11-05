"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _value(h):
    """Compute the objective value. Lower is better."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit with safety margin
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # Should be 2.0
    
    # Try several different initial patterns
    initial_patterns = []
    
    # Pattern 1: All ones (seed)
    initial_patterns.append(np.ones(N))
    
    # Pattern 2: Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1.0
    initial_patterns.append(h_alt)
    
    # Pattern 3: Triangular / ramp
    h_ramp = np.linspace(1, -1, N)
    initial_patterns.append(h_ramp)
    
    # Pattern 4: Two halves positive, two halves negative
    h_half = np.ones(N)
    h_half[N//2:] = -1.0
    initial_patterns.append(h_half)
    
    # Pattern 5: Sine wave
    h_sine = np.sin(2 * np.pi * np.arange(N) / N)
    initial_patterns.append(h_sine)
    
    # Pattern 6: Cosine wave
    h_cos = np.cos(2 * np.pi * np.arange(N) / N)
    initial_patterns.append(h_cos)
    
    # Pattern 7: Random
    np.random.seed(42)
    h_rand = np.random.randn(N)
    initial_patterns.append(h_rand)
    
    # Pattern 8: Sparse - mostly zeros with a few ones
    h_sparse = np.zeros(N)
    h_sparse[::10] = 1.0
    initial_patterns.append(h_sparse)
    
    # Pattern 9: Block structure
    h_block = np.ones(N)
    h_block[100:200] = -1.0
    initial_patterns.append(h_block)
    
    # Pattern 10: Quadratic
    t = np.linspace(-1, 1, N)
    h_quad = 1 - t**2
    initial_patterns.append(h_quad)
    
    # Evaluate all initial patterns
    for h0 in initial_patterns:
        if time.time() > deadline - 5:
            break
        val = _value(h0)
        if val < best_val:
            best_val = val
            best_h = h0.copy()
    
    # Try local optimization from the best initial pattern
    def objective(h):
        if time.time() > deadline - 2:
            return best_val  # Stop optimizing
        return _value(h)
    
    # Use Nelder-Mead for derivative-free optimization
    # We'll try a few restarts with different initial conditions
    for i, h0 in enumerate(initial_patterns[:5]):  # Try first 5 patterns
        if time.time() > deadline - 10:
            break
        
        # Normalize to avoid numerical issues
        s = np.sum(h0)
        if abs(s) < 1e-10:
            continue
        h_start = h0 / s  # Normalize so sum is 1
        
        try:
            result = minimize(
                objective,
                h_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Final check
    if time.time() > deadline - 1:
        return best_h
    
    # One more pass with the best found so far
    if abs(np.sum(best_h)) > 1e-10:
        h_start = best_h / np.sum(best_h)
        try:
            result = minimize(
                objective,
                h_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
