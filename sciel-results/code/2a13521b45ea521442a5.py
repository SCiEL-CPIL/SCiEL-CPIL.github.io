"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a local baseline estimated from the GP posterior to target
    boundaries of explored basins, combined with uncertainty weighting.
    """
    start_time = time.time()
    deadline = 300.0  # seconds per evaluation

    n = len(mu)
    eps = 1e-9

    # Ensure finite values
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)

    # Standard z-score for EI
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (standard)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Local baseline: use the GP mean itself as a proxy for local f
    # The idea: improvement over local mean is (mu_local - mu)
    # But we want to target where variance is high relative to local prediction.
    
    # Uncertainty-Weighted Improvement (UWI):
    # Reward candidates where sigma is high relative to the distance from f_best
    # This encourages exploration in uncertain regions that might contain better solutions
    
    # Normalize sigma to [0, 1] based on observed range
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range < eps:
        sigma_norm = np.ones_like(sigma_safe)
    else:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range

    # Combine EI and uncertainty
    # Higher sigma_norm means more uncertainty, which we reward
    # But we also need the EI to be meaningful
    
    # Use a weighted combination: EI * (1 + alpha * sigma_norm)
    # This boosts exploration in uncertain regions
    alpha = 0.5
    uwi = ei * (1.0 + alpha * sigma_norm)

    # Cost-aware: divide by cost
    scores = uwi / cost_safe

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline check
    if time.time() - start_time > deadline - 0.1:
        return scores

    return scores
# EVOLVE-BLOCK-END
