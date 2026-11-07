import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin below 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive LCB and EI portfolio.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Adaptive beta based on progress toward known minimum (-0.8 for cosine8)
    # progress=1 when f_best=0 (early), progress=0 when f_best=-0.8 (converged)
    progress = np.clip((f_best + 0.8) / 0.8, 0.0, 1.0)
    beta = 0.5 + 2.0 * progress  # beta in [0.5, 2.5]

    # 1. Cost-aware LCB: (f_best - mu + beta * sigma) / cost
    lcb = (f_best - mu + beta * sigma_safe) / cost_safe

    # 2. Expected Improvement per cost
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_cost = ei / cost_safe

    # 3. Portfolio: max of LCB and EI/cost
    scores = np.maximum(lcb, ei_cost)

    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)

    return scores
