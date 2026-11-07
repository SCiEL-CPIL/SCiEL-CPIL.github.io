"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with small exploitation bias
    xi = 0.05
    improvement = f_best - xi - mu
    z = improvement / (sigma + 1e-9)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    # Clip negative values to zero to avoid negative scores due to numerical noise
    ei = np.maximum(ei, 0.0)
    return ei / (cost + 1e-9)
# EVOLVE-BLOCK-END
