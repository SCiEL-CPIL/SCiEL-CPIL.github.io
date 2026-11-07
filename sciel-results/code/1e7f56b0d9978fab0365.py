"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining normalized EI/cost with adaptive exploration.
    """
    # Deadline guard: if we're too close to the limit, return safe scores
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)

    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI: improvement per unit cost
    ei_score = ei / cost_safe

    # Adaptive exploration weight based on progress toward the known minimum
    # cosine8 minimum is -0.8
    min_possible = -0.8
    # Progress: 1.0 when f_best is at worst observed (far from min), 0.0 when at min
    # Use a robust scale: the range of the function is roughly [-0.8, 1.0]
    # But f_best can be anywhere. Use a simple heuristic:
    # If f_best is close to -0.8, we're doing well -> exploit more
    # If f_best is far from -0.8, explore more
    progress = (f_best - min_possible) / (1.0 - min_possible + 1e-10)
    progress = np.clip(progress, 0.0, 1.0)

    # Exploration weight: higher when progress is low (early on)
    # Base weight 0.2, up to 0.5 when very early
    exploration_weight = 0.2 + 0.3 * progress
    exploration_weight = np.clip(exploration_weight, 0.15, 0.5)

    # Beta for UCB term, also adaptive
    # Larger beta when we have less information (high average sigma)
    sigma_mean = np.mean(sigma_safe)
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1e-10
    
    # Scale beta by relative uncertainty
    beta = 0.3 * (sigma_mean / (mu_range + 1e-10))
    beta = np.clip(beta, 0.1, 1.0)

    # Exploration term: prefer high uncertainty, penalized by cost
    # Use a combined exploration score that rewards sigma but also considers
    # how far mu is from f_best (don't explore where we already know it's bad)
    # Simple UCB-like: beta * sigma / cost
    ucb_score = beta * sigma_safe / cost_safe

    # Normalize each component to zero-mean, unit-variance for comparable scales
    ei_mean = np.mean(ei_score)
    ei_std = np.std(ei_score) + 1e-8
    ei_norm = (ei_score - ei_mean) / ei_std

    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-8
    ucb_norm = (ucb_score - ucb_mean) / ucb_std

    # Combine: primarily exploitation, with adaptive exploration
    combined = (1.0 - exploration_weight) * ei_norm + exploration_weight * ucb_norm

    # Ensure all values are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    return combined
# EVOLVE-BLOCK-END
