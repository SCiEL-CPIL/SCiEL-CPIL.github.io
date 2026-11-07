import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) divided by cost.
    EI is the standard, well-calibrated choice for BO; dividing by cost makes it cost-aware.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    # Standard EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Numerical safety: clip tiny negative values from floating point error
    ei = np.maximum(ei, 0.0)
    # Cost-aware: divide by cost to prefer cheaper points
    scores = ei / cost
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
