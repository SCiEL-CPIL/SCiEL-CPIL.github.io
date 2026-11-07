import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI).
    PI is more robust to scale changes in f_best compared to EI,
    which can vanish in the late stages of optimization.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Probability that f(x) < f_best
    pi = norm.cdf(z)
    # Ensure non-negative
    pi = np.maximum(pi, 0.0)
    # Cost-aware: divide by cost
    return pi / cost
