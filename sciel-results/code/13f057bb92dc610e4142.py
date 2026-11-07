"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Draw samples from the posterior predictive distribution
    # Note: Relies on the global NumPy random state managed by the evaluator
    z = np.random.randn(mu.shape[0])
    return mu + sigma * z
# EVOLVE-BLOCK-END
