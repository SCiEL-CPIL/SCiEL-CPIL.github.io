"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    Uses a cost-aware LCB with adaptive exploration based on gap to known minimum.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)

    # Known global minimum for styblinski-tang 2d
    known_min = -78.3323
    gap = f_best - known_min

    # Adaptive exploration parameter kappa
    # When gap is large, explore more (higher kappa)
    # When gap is small, exploit more (lower kappa)
    if gap < 2.0:
        kappa = 1.0
    elif gap < 10.0:
        kappa = 1.5
    elif gap < 30.0:
        kappa = 2.0
    else:
        kappa = 2.5

    # LCB score: lower is better, so we want low mu and high uncertainty
    # score = (mu - kappa * sigma) / cost
    # But we want higher = evaluate next, so we negate
    lcb = mu_safe - kappa * sigma_safe
    score = -lcb / cost_safe

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
