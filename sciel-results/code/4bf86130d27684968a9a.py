"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 280.0
_start_time = time.time()


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition for minimizing rastrigin 2d.
    
    Uses a lower confidence bound: score = -mu - beta * sigma, normalized by cost.
    The beta coefficient adapts to the current uncertainty level in the GP,
    providing stronger exploration when the model is uncertain and weaker
    exploitation when it's confident.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: scale exploration by the relative uncertainty of the GP.
    # When sigma is large relative to the spread of mu, we're in unexplored regions.
    # Use the median sigma as a reference for "typical" uncertainty.
    sigma_med = np.median(sigma_safe)
    if sigma_med < 1e-8:
        sigma_med = 1e-8
    
    # Relative uncertainty: how much more uncertain is each point vs. typical?
    rel_uncert = sigma_safe / sigma_med
    
    # Beta: base exploration plus a boost for highly uncertain points.
    # The base beta is tuned for Rastrigin's multimodality.
    # As f_best improves (gets closer to 0), we can afford less exploration.
    # f_best starts high (~10-20) and decreases toward 0.
    progress = 1.0 - np.clip(f_best / 20.0, 0.0, 1.0)
    
    # Base beta: higher early, lower late
    beta_base = 2.0 * (1.0 - 0.5 * progress)
    
    # Additional exploration boost for points with above-median uncertainty
    # This helps escape local minima in Rastrigin
    boost = 0.5 * np.maximum(rel_uncert - 1.0, 0.0)
    
    beta = beta_base + boost
    
    # LCB: we want to minimize, so low mu and high sigma are good.
    # score = -mu - beta * sigma (higher is better)
    lcb = -mu - beta * sigma_safe
    
    # Normalize by cost: prefer cheaper evaluations
    # Use sqrt(cost) to soften the penalty
    cost_penalty = np.sqrt(cost_safe)
    scores = lcb / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
