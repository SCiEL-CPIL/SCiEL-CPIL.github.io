"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI).
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / (sigma + 1e-9)
    return norm.cdf(z)
# EVOLVE-BLOCK-END
