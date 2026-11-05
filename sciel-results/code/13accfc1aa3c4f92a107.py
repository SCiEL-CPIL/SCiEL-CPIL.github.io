"""
Strategy: Construct a real-valued sequence inspired by Zadoff-Chu properties
(constant amplitude, low autocorrelation sidelobes) and refine it using
gradient-based optimization with a smooth max surrogate to minimize the
autocorrelation inequality value.
"""
import numpy as np
import scipy.optimize as so
import time

N = 400

def compute_value(h):
    """
    Computes the task objective value.
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    """
    s = np.sum(h)
    if np.abs(s) < 1e-12:
        return 1e12
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2 * N * max_conv / (s * s))
    return val

def smooth_max(x, beta=100.0):
    """
    Log-sum-exp approximation of max(x).
    """
    # Use a stable version of log-sum-exp for max
    # max(x) = (1/beta) * log(sum(exp(beta * x)))
    # For numerical stability, shift x
    x_shifted = x - np.max(x)
    return np.max(x) + np.log(np.sum(np.exp(beta * x_shifted))) / beta

def objective(params, beta=100.0):
    """
    Objective function for the optimizer.
    params: 400 real values.
    """
    h = params
    s = np.sum(h)
    if np.abs(s) < 1e-12:
        return 1e12
    conv = np.convolve(h, h)
    # Use smooth max to allow gradients
    # Note: The original value uses hard max. We use smooth max for optimization.
    max_conv_smooth = smooth_max(conv, beta)
    val = abs(2 * N * max_conv_smooth / (s * s))
    return val

def construct():
    start_time = time.time()
    time_limit = 115  # Leave margin for 120s limit
    
    # 1. Generate initial candidates
    candidates = []
    
    # Candidate 1: Seed (fallback)
    h_seed = np.ones(N)
    candidates.append(h_seed)
    
    # Candidate 2: Zadoff-Chu inspired (Real-valued CAZAC-like)
    # A real-valued sequence with low autocorrelation can be approximated by
    # taking the real part of a complex ZC sequence or using a m-sequence.
    # Let's use a simple m-sequence (pseudo-random binary +/- 1) as a starting point
    # for optimization, as it has good autocorrelation properties.
    rng = np.random.default_rng(42)
    h_mseq = rng.choice([-1, 1], size=N)
    # Scale to have similar magnitude as seed? No, scaling is invariant.
    # But we need sum != 0.
    if np.abs(np.sum(h_mseq)) < 1.0:
        h_mseq += 0.1
    candidates.append(h_mseq)

    # Candidate 3: Random Gaussian
    h_rand = rng.standard_normal(N)
    candidates.append(h_rand)
    
    # Candidate 4: Sine wave (often has low energy in high lags)
    t = np.linspace(0, 1, N)
    h_sine = np.sin(2 * np.pi * t) + 0.5  # Shift to ensure positive sum
    candidates.append(h_sine)

    # Evaluate initial candidates
    best_h = candidates[0]
    best_val = compute_value(best_h)
    
    for h in candidates[1:]:
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h

    # 2. Optimization Phase
    # Use L-BFGS-B to refine the best candidate
    # We optimize the heights directly.
    
    current_h = best_h.copy()
    current_val = best_val
    
    # Define bounds? Heights can be any real. No bounds.
    # However, to avoid overflow, we might want to constrain magnitude?
    # The objective is scale invariant, so the optimizer might drift.
    # We can fix the sum to 1.0 to remove the scale degree of freedom.
    # Let's re-parameterize: h = sum_target * normalized_h
    # Or just constrain one variable?
    # Easier: Optimize h, but if sum becomes small, penalty.
    # Actually, since it's scale invariant, the optimizer might struggle with the scale.
    # Let's constrain the L2 norm or Sum to be 1.
    # If we fix sum(h) = 1, then value = 2 * N * max(conv(h,h)).
    # This is a simpler problem.
    
    # Let's reformulate:
    # Minimize 2 * N * max(conv(h, h)) subject to sum(h) = 1.
    # This removes the denominator variation.
    
    def objective_normalized(params):
        # params are the heights h
        # We enforce sum(h) = 1 by projecting or penalizing
        s = np.sum(params)
        # Soft constraint: add penalty if sum != 1
        penalty = (s - 1.0) ** 2 * 100.0
        conv = np.convolve(params, params)
        max_conv = smooth_max(conv, beta=100.0)
        return 2.0 * N * max_conv + penalty

    # Initial guess: Normalize best_h to sum 1
    h0 = best_h / np.sum(best_h)
    
    # Run L-BFGS-B
    # Time budget check
    if time.time() - start_time < time_limit - 10:
        try:
            res = so.minimize(objective_normalized, h0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < compute_value(res.x):
                # Check validity
                h_opt = res.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val:
                    best_h = h_opt
                    best_val = val_opt
        except Exception:
            pass

    # 3. Final Polish (High precision check)
    # If we have time, try a few random restarts around the best solution
    if time.time() - start_time < time_limit - 5:
        for _ in range(3):
            h_trial = best_h + rng.standard_normal(N) * 0.01
            # Normalize to sum 1
            h_trial = h_trial / np.sum(h_trial)
            
            def obj_polk(params):
                s = np.sum(params)
                penalty = (s - 1.0) ** 2 * 100.0
                conv = np.convolve(params, params)
                max_conv = smooth_max(conv, beta=200.0) # Higher beta for sharper max
                return 2.0 * N * max_conv + penalty
                
            res = so.minimize(obj_polk, h_trial, method='L-BFGS-B',
                              options={'maxiter': 50, 'ftol': 1e-12})
            if res.success:
                val_trial = compute_value(res.x)
                if val_trial < best_val:
                    best_h = res.x
                    best_val = val_trial
            if time.time() - start_time > time_limit - 2:
                break

    return best_h

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to ensure the entry point is just construct().
# The EVOLVE block usually contains the code to be evolved.
# I will place the entire logic inside the EVOLVE block as per the prompt structure.
# EVOLVE-BLOCK-END
