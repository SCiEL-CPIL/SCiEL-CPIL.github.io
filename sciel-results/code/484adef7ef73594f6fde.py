import numpy as np
import time

# Deadline guard
TIME_BUDGET = 320.0
START_TIME = None

def _elapsed():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME

def acquisition(mu, sigma, f_best):
    """
    Adaptive hybrid acquisition function blending UCB (exploration) and
    pure exploitation (low mu) for minimization of Michalewicz 2D.
    
    The weight shifts from exploration to exploitation as f_best improves
    toward the known global minimum (-1.8013).
    
    Returns 2000 finite scores, higher meaning evaluate next.
    
    Parameters:
    mu: array of shape (N,), posterior mean in objective units.
    sigma: array of shape (N,), posterior standard deviation in objective units.
    f_best: float, current best observed value.
    
    Returns:
    array of shape (N,), finite scores.
    """
    # Deadline guard
    if _elapsed() > TIME_BUDGET:
        # Fallback: Simple UCB with small beta
        beta = 0.1
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = beta * sigma_safe - mu
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known global minimum of Michalewicz 2D
    f_min = -1.8013
    
    # Adaptive beta: start high for exploration, decrease as f_best improves
    # When f_best is far from f_min, explore more (higher beta)
    # When f_best is close to f_min, exploit more (lower beta)
    
    # Normalize progress: 0 when f_best is at initial random level, 1 when at f_min
    # Initial random points might be around 0 to 5, so use a reasonable range
    f_initial = 2.0  # Rough estimate of typical initial function values
    f_range = max(f_initial - f_min, 1e-6)
    progress = np.clip((f_initial - f_best) / f_range, 0.0, 1.0)
    
    # Beta decreases from ~2.0 (heavy exploration) to ~0.3 (heavy exploitation)
    beta_start = 2.0
    beta_end = 0.3
    beta = beta_start + (beta_end - beta_start) * progress
    
    # Calculate UCB score for maximization
    # We want to find low values of f.
    # Lower bound of f is approx mu - beta * sigma.
    # We maximize the potential to be low, so we maximize - (mu - beta * sigma)
    # which is beta * sigma - mu.
    scores = beta * sigma_safe - mu_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
