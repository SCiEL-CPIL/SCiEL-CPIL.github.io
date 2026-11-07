"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Normalized Expected Improvement (NEI).
    
    Strategy:
    - Computes Expected Improvement (EI) for minimization.
    - Normalizes by the standard deviation (sigma) to create a risk-adjusted score.
    - This encourages points that have a high expected improvement relative to their uncertainty.
    - Includes a deadline guard to ensure the function returns within the time limit.
    """
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # Safety margin: if we are close to the deadline, return a fast greedy score
    if remaining_time < _DEADLINE_MARGIN:
        # Greedy: pick points with lowest mu (highest probability of being good)
        # Higher score is better, so we invert mu
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    eps = 1e-9
    sigma_safe = sigma + eps

    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe

    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Calculate Normalized EI (NEI)
    # NEI = EI / sigma
    # This scales the improvement by the uncertainty, favoring confident improvements
    nei = ei / sigma_safe

    # Ensure all scores are finite
    # If sigma is very small, NEI can be large, which is good (exploit known good points)
    # If ei is negative (mu > f_best), NEI is negative (avoid bad points)
    scores = np.where(np.isfinite(nei), nei, -np.inf)

    return scores
# EVOLVE-BLOCK-END
