"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses a hybrid approach combining Expected Improvement (EI) and Upper Confidence
    Bound (UCB) with adaptive exploration/exploitation weighting based on f_best.
    
    For the multimodal Rastrigin landscape, the exploration weight is tuned to help
    escape local minima while exploiting cheap promising points.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Guard against numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    improvement = f_best - mu  # positive when mu < f_best
    z = improvement / sigma_safe
    
    # Expected Improvement (positive when mu < f_best)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB term: mu - beta * sigma (we want to minimize, so lower mu is better)
    # For minimization, UCB = mu - beta * sigma
    # Higher UCB score means more likely to be good
    # We use a beta that adapts based on f_best
    if f_best > 10.0:
        beta = 2.5  # high exploration early on
    elif f_best > 5.0:
        beta = 1.5
    elif f_best > 1.0:
        beta = 0.8
    else:
        beta = 0.3  # low exploration when close to optimum
    
    ucb = mu - beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ucb_max = np.max(np.abs(ucb)) if np.max(np.abs(ucb)) > 1e-10 else 1.0
    ei_norm = ei / ei_max
    ucb_norm = -ucb / ucb_max  # negate because lower ucb is better for minimization
    
    # Adaptive blend: more EI when f_best is low (exploitation), more UCB when high (exploration)
    if f_best > 10.0:
        ei_weight = 0.3
        ucb_weight = 0.7
    elif f_best > 5.0:
        ei_weight = 0.5
        ucb_weight = 0.5
    elif f_best > 1.0:
        ei_weight = 0.7
        ucb_weight = 0.3
    else:
        ei_weight = 0.85
        ucb_weight = 0.15
    
    # Combined acquisition before cost normalization
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Cost-aware score: divide by cost
    scores = combined / cost_safe
    
    # Add a small exploration bonus for early iterations
    if f_best > 5.0:
        explore_weight = min(1.0, (f_best - 5.0) / 15.0) * 0.15
        explore_term = explore_weight * sigma_safe / cost_safe
        scores = scores + explore_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
