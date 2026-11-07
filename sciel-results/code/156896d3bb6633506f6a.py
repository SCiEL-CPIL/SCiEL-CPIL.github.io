"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time and enforce a safety margin
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    elapsed = _get_elapsed()
    # Safety margin: if we're getting close to the 330s limit, return a simple score
    if elapsed > 320:
        # Return a simple score based on expected improvement without heavy computation
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * 0.5 * (1 + np.tanh(z)) + sigma * np.exp(-0.5 * z**2)) / cost
    
    # Use a robust acquisition function: PI-like with cost awareness
    # PI is more robust than EI for short budgets
    # Use a threshold-based approach to balance exploration and exploitation
    
    # Compute z-score
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Probability of improvement (PI)
    # P(mu < f_best) = Phi((f_best - mu) / sigma)
    # Use a numerically stable approximation
    pi = 0.5 * (1 + np.tanh(0.7978845608028654 * z))  # tanh approximation to normal CDF
    
    # Expected improvement component (for exploitation)
    # E[max(f_best - mu, 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But for short budgets, PI alone can be more robust
    # Blend PI with a small EI component
    ei = (f_best - mu) * pi + sigma * np.exp(-0.5 * z**2)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # But use a softer cost scaling to avoid over-penalizing expensive points
    cost_factor = cost ** 0.5  # Softer than linear cost penalty
    
    # Blend: primarily PI-based, with some EI for exploitation
    score = (0.8 * pi + 0.2 * ei / (sigma + 1e-9)) / cost_factor
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
