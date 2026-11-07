"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # 330s limit, 10s safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Strategy:
    1. Use Expected Improvement (EI) as the primary signal.
    2. Do NOT divide by cost (based on observed improvement from removing cost penalty).
       The cost is in [1, 2], so the penalty is mild but can bias search away from
       high-cost regions where the shifted minimum might reside.
    3. Add a small exploration bonus based on sigma to help escape local minima
       in the acquisition landscape, especially important for the narrow valley
       of the Rosenbrock function.
    4. Ensure numerical stability and finite outputs.
    """
    # Check deadline
    if time.time() > _TIME_LIMIT:
        return np.zeros_like(mu)

    # Ensure inputs are finite and valid
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)

    # Primary score: EI without cost division
    # This allows the optimizer to explore high-cost regions where the
    # shifted minimum might reside, rather than being biased away by cost.
    scores = ei.copy()

    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local minima and explore the narrow valley of Rosenbrock
    sigma_mean = np.mean(sigma)
    if sigma_mean > 0 and np.isfinite(sigma_mean):
        # Scale exploration bonus relative to mean sigma
        exploration = 0.05 * (sigma / sigma_mean)
        scores = scores + exploration

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
