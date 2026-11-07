"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Robust Expected Improvement with noise floor to avoid over-exploration
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Add a small UCB-like term to encourage exploration in early stages
    # but keep it minimal to not disrupt the pure EI behavior
    ucb_term = 0.01 * sigma_safe
    return ei + ucb_term
# EVOLVE-BLOCK-END
