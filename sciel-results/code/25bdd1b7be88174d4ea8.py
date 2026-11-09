"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses an adaptive blend of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    with exploration weight that increases as f_best gets closer to 0 (the minimum),
    helping escape local minima in the multimodal Rastrigin landscape.
    
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
    # Guard against division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (positive when mu < f_best)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB: balance exploitation (mu) and exploration (sigma)
    # Use beta that adapts based on how close f_best is to the known minimum (0)
    # When f_best is large (far from minimum), explore more; when close, exploit more
    # f_best for Rastrigin ranges from ~20+ (initial) down to 0 (optimum)
    # Normalize: f_best / 20 gives a measure of remaining distance
    f_best_norm = np.clip(f_best / 20.0, 0.0, 1.0)
    
    # Beta: higher when f_best is large (more exploration needed)
    # beta ranges from 0.5 (exploit) to 2.0 (explore)
    beta = 0.5 + 1.5 * f_best_norm
    
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB, both normalized by cost
    # EI is already in objective units; UCB is also in objective units
    # Weight: use 0.6 * EI + 0.4 * UCB as a base blend
    # But adapt the blend: when f_best is high, weight UCB more
    ei_weight = 0.7 - 0.3 * f_best_norm  # 0.7 when f_best~0, 0.4 when f_best~20
    ucb_weight = 1.0 - ei_weight          # 0.3 when f_best~0, 0.6 when f_best~20
    
    combined = ei_weight * ei + ucb_weight * ucb
    
    # Divide by cost to get cost-aware score
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
