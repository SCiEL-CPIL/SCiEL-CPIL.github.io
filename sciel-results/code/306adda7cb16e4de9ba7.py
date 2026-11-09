"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) per unit cost.
    # PI is robust for exploration in sparse, multimodal functions like Griewank.
    # It avoids the potential instability of EI when sigma is very large or small.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    # Higher score = better
    scores = pi / cost
    return scores
# EVOLVE-BLOCK-END
