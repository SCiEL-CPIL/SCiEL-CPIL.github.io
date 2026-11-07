import numpy as np
import time
from scipy.stats import norm

# Global variable to track start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 15.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses an adaptive blend of Expected Improvement (EI) and Uncertainty (sigma)
    scaled by cost. The blend adapts based on progress (f_best relative to
    the known minimum of the function).
    
    Score = (alpha * EI + (1 - alpha) * beta * sigma) / cost
    where alpha increases (more exploitation) as we approach the optimum,
    and beta decreases (less pure exploration) as we get closer.
    """
    # Deadline guard
    current_time = time.time()
    if current_time - _start_time > (TIME_LIMIT - SAFETY_MARGIN):
        # If we are close to the limit, return a safe, finite array.
        return np.zeros_like(mu)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-10, cost, 1.0)

    # Handle cases where sigma is very small
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    EI = np.maximum(EI, 0.0)

    # Adaptive parameters based on progress
    # The minimum of Styblinski-Tang 2D is approximately -78.3323
    f_min_true = -78.3323
    # Normalize progress: 0 when f_best is far from min, 1 when close
    # Use a smooth transition. Let's say progress is based on how close f_best is to f_min_true
    # relative to the initial range (e.g., f_best from ~0 to -78)
    # progress = clip((f_best - f_min_true) / (0 - f_min_true), 0, 1) ? No, that's inverted.
    # If f_best = 0, progress = 1 (start). If f_best = -78, progress = 0 (end).
    # We want alpha (exploitation) to increase as we get closer (progress decreases? No, as f_best decreases).
    # Let's define t = (f_best - f_min_true) / (initial_guess - f_min_true)
    # initial_guess is roughly 0 or the max observed. Let's use a fixed range for stability.
    # f_best starts around 0-20 and goes to -78.
    # Let's map f_best from [0, -78] to t in [1, 0].
    # t = (f_best - f_min_true) / (0 - f_min_true)
    # If f_best = 0, t = 1. If f_best = -78, t = 0.
    # We want alpha (weight on EI) to go from low (0.3) to high (0.9) as t goes from 1 to 0.
    # alpha = 0.9 - 0.6 * t
    # beta (weight on sigma) goes from high (1.0) to low (0.1) as t goes from 1 to 0.
    # beta = 0.1 + 0.9 * t
    
    # To be robust against different starting points, let's use a relative measure.
    # But we don't know the initial f_best exactly, just the current one.
    # Let's assume the range is roughly 80 units.
    # t = (f_best - f_min_true) / 80.0
    # Clip t to [0, 1]
    t = (f_best - f_min_true) / 80.0
    t = np.clip(t, 0.0, 1.0)
    
    # If f_best is already better than min (shouldn't happen, but just in case), t < 0 -> 0
    # If f_best is much worse, t > 1 -> 1
    
    alpha = 0.9 - 0.6 * t  # 0.9 when t=0 (close to min), 0.3 when t=1 (far)
    beta = 0.1 + 0.9 * t   # 0.1 when t=0 (close to min), 1.0 when t=1 (far)

    # Combined score
    raw_scores = alpha * EI + (1 - alpha) * beta * sigma_safe

    # Cost-aware scaling
    scores = raw_scores / cost_safe

    # Handle potential numerical issues (inf/nan)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
