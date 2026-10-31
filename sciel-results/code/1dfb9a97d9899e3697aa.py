"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    # Avoid overflow
    if alpha * (np.max(x) - x_max) > 700:
        return x_max
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _project(h):
    """Project h onto the set {x : sum(x) = N/2, 0 <= x <= 1}."""
    # Simple iterative projection (clipping and adjusting sum)
    h = h.copy()
    for _ in range(100):
        # Clip to bounds
        h_clipped = np.clip(h, 0.0, 1.0)
        # Calculate difference in sum
        diff = N / 2.0 - np.sum(h_clipped)
        if abs(diff) < 1e-12:
            break
        # Distribute difference to non-saturated variables
        # Variables that are not at 0 or 1 can be adjusted
        mask = (h_clipped > 1e-6) & (h_clipped < 1.0 - 1e-6)
        if not np.any(mask):
            break
        h_clipped[mask] += diff / np.sum(mask)
        h = h_clipped
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Initial guess: A sawtooth pattern is often better than a 2-block pattern
    # We want h[i] and h[i+1] to be "opposite" to minimize h[i](1-h[i+1])
    # If h[i] is high, h[i+1] should be low.
    # A simple alternating pattern 0.5, 0.5 gives overlap 0.25.
    # A pattern like 0.8, 0.2, 0.8, 0.2 gives overlap 0.8*0.8 = 0.64 (bad).
    # Wait, overlap is h[i] * (1 - h[i+1]).
    # If h[i]=0.8, h[i+1]=0.2, overlap = 0.8 * 0.8 = 0.64.
    # If h[i]=0.2, h[i+1]=0.8, overlap = 0.2 * 0.2 = 0.04.
    # So we want h[i] high when h[i+1] is high? No.
    # We want to minimize max(h[i] * (1 - h[i+1])).
    # This is minimized when h[i] and h[i+1] are both high or both low?
    # If both high (e.g., 0.8, 0.8), overlap = 0.8 * 0.2 = 0.16.
    # If both low (e.g., 0.2, 0.2), overlap = 0.2 * 0.8 = 0.16.
    # If alternating (0.8, 0.2), overlap = 0.8 * 0.8 = 0.64.
    # So we want h to be "smooth" or "clustered", not alternating.
    # The 2-block solution (A, A, ..., B, B) is clustered.
    # Let's try a triangular wave that goes up and down smoothly.
    
    # Generate a triangular wave
    x = np.linspace(0, 2, N, endpoint=False)
    # Triangle wave with period 1
    h0 = np.abs(2 * (x % 1.0) - 1.0)
    # Scale to have sum N/2
    # Sum of triangle wave is approx N/2
    # Let's just normalize
    h0 = h0 / np.sum(h0) * (N / 2.0)
    h0 = _project(h0)
    
    # Another good initial guess: The 2-block solution
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h1 = np.zeros(N)
    half = N // 2
    h1[:half] = A
    h1[half:] = B
    h1 = _project(h1)

    current_score = min(_score(h0), _score(h1))
    h_final = h0 if _score(h0) < _score(h1) else h1

    # Optimization loop with multiple restarts
    best_h = h_final.copy()
    best_score = current_score

    # Define the objective function for the optimizer
    # We minimize t + penalty * (smooth_max(overlap) - t)^2
    # This is a smooth approximation of min t s.t. overlap_i <= t
    
    alpha = 50.0  # Smoothness parameter
    
    def objective(params):
        h = params[:N]
        t = params[N]
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=alpha)
        # Penalty for constraint violation
        penalty = 100.0 * np.maximum(0.0, smax - t)**2
        # Also add a small penalty for sum constraint violation to help convergence
        sum_penalty = 10.0 * (np.sum(h) - N/2.0)**2
        return t + penalty + sum_penalty

    def jac(params):
        # Numerical gradient is safer for this complex objective
        # But L-BFGS-B can use numerical gradient if not provided
        # Let's rely on numerical gradient for simplicity and robustness
        return None

    # Bounds for h are [0, 1], for t is [0, 1]
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]

    # We will run several restarts with different initial guesses
    # 1. The best found so far
    # 2. Random perturbations of the best found so far
    # 3. Other structured patterns

    initial_guesses = [best_h.copy()]
    
    # Add random perturbations
    np.random.seed(42)
    for i in range(5):
        h_rand = best_h + np.random.normal(0, 0.1, N)
        h_rand = _project(h_rand)
        initial_guesses.append(h_rand)
        
    # Add a "smooth" random pattern
    for i in range(3):
        # Generate a low-frequency noise
        freqs = np.random.uniform(0.1, 2.0, 5)
        phases = np.random.uniform(0, 2*np.pi, 5)
        h_smooth = np.zeros(N)
        for f, p in zip(freqs, phases):
            h_smooth += np.sin(2 * np.pi * f * np.arange(N) / N + p)
        h_smooth = (h_smooth - np.min(h_smooth)) / (np.max(h_smooth) - np.min(h_smooth) + 1e-9)
        h_smooth = h_smooth * 0.8 + 0.1  # Keep in [0.1, 0.9] roughly
        h_smooth = _project(h_smooth)
        initial_guesses.append(h_smooth)

    for h_init in initial_guesses:
        if time.time() > deadline - 10.0:
            break
        
        x0 = np.concatenate([h_init, [best_score + 0.1]])
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score + 0.01:
                h_opt = result.x[:N].copy()
                h_opt = _project(h_opt)
                
                new_score = _score(h_opt)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_opt.copy()
                    
        except Exception:
            pass

    # Final polish with higher alpha and tighter tolerances
    if time.time() < deadline - 5.0:
        alpha_high = 200.0
        
        def objective_polish(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=alpha_high)
            penalty = 1000.0 * np.maximum(0.0, smax - t)**2
            sum_penalty = 100.0 * (np.sum(h) - N/2.0)**2
            return t + penalty + sum_penalty

        x0_polish = np.concatenate([best_h, [best_score + 0.01]])
        
        try:
            result_polish = minimize(
                objective_polish,
                x0_polish,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result_polish.success:
                h_opt_polish = result_polish.x[:N].copy()
                h_opt_polish = _project(h_opt_polish)
                
                new_score_polish = _score(h_opt_polish)
                if new_score_polish < best_score:
                    best_score = new_score_polish
                    best_h = h_opt_polish.copy()
                    
        except Exception:
            pass

    return best_h
