"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _compute_value(h):
    """Compute the task's objective value."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))


def _objective(x):
    """Objective function for optimization: minimize the value."""
    h = x
    s = np.sum(h)
    if s == 0:
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2 * N * np.max(conv) / (s ** 2))
    return val


def construct():
    """Construct step heights minimizing the autocorrelation inequality value."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # Should be 2.0
    
    # Strategy: Use a non-symmetric sequence to break the value=2 barrier.
    # The key insight is that for non-symmetric sequences, the maximum of the
    # autoconvolution can be reduced below the Cauchy-Schwarz bound.
    
    # Initialize with a structured non-symmetric pattern
    # Try various initial patterns and optimize
    
    # Pattern 1: Linear ramp (asymmetric)
    h_ramp = np.linspace(-1, 1, N)
    h_ramp = h_ramp - np.mean(h_ramp)  # Center it
    if abs(np.sum(h_ramp)) > 1e-10:
        val_ramp = _compute_value(h_ramp)
        if val_ramp < best_val:
            best_val = val_ramp
            best_h = h_ramp.copy()
    
    # Pattern 2: Step function with negative left, positive right
    h_step = np.zeros(N)
    h_step[:N//2] = -1.0
    h_step[N//2:] = 1.0
    val_step = _compute_value(h_step)
    if val_step < best_val:
        best_val = val_step
        best_h = h_step.copy()
    
    # Pattern 3: Smooth asymmetric (e.g., exponential decay on one side)
    x = np.linspace(-1, 1, N)
    h_exp = np.exp(-2 * x) - np.mean(np.exp(-2 * x))
    val_exp = _compute_value(h_exp)
    if val_exp < best_val:
        best_val = val_exp
        best_h = h_exp.copy()
    
    # Pattern 4: Sine-based asymmetric
    h_sine = np.sin(np.pi * x) * np.cos(np.pi * x) - np.mean(np.sin(np.pi * x) * np.cos(np.pi * x))
    val_sine = _compute_value(h_sine)
    if val_sine < best_val:
        best_val = val_sine
        best_h = h_sine.copy()
    
    # Pattern 5: Random asymmetric with some structure
    np.random.seed(42)
    for trial in range(5):
        h_rand = np.random.randn(N)
        # Make it asymmetric by applying a linear trend
        h_rand = h_rand + 0.5 * np.linspace(-1, 1, N)
        h_rand = h_rand - np.mean(h_rand)
        val_rand = _compute_value(h_rand)
        if val_rand < best_val:
            best_val = val_rand
            best_h = h_rand.copy()
    
    # Now use gradient-based optimization to refine
    # Use L-BFGS-B which is efficient for large problems
    h0 = best_h.copy()
    
    # Check if we have time for optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            result = minimize(
                _objective,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 5000, 'maxls': 50}
            )
            if result.success or result.fun < best_val:
                h_opt = result.x
                val_opt = _compute_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try another optimization from a different starting point
        try:
            # Start from a different pattern
            h0_2 = np.zeros(N)
            h0_2[:N//4] = -0.5
            h0_2[N//4:3*N//4] = 0.5
            h0_2[3*N//4:] = -0.3
            h0_2 = h0_2 - np.mean(h0_2)
            
            result2 = minimize(
                _objective,
                h0_2,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 3000, 'maxls': 50}
            )
            if result2.success or result2.fun < best_val:
                h_opt2 = result2.x
                val_opt2 = _compute_value(h_opt2)
                if val_opt2 < best_val:
                    best_val = val_opt2
                    best_h = h_opt2.copy()
        except Exception:
            pass
    
    # Final check: ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
