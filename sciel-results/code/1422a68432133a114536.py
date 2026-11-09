"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement with a cost-regularization that favors the inner domain.
    # The valid minimizer region corresponds to cost in [1.1, 1.9].
    # Corners (cost 1.0 and 2.0) are penalized to avoid wasting budget on invalid regions.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost penalty: Gaussian centered at 1.5 (center of the domain/cost range)
    # This suppresses the corners (cost ~1.0 and ~2.0) which are outside the inner 80%
    # where the global minimum is guaranteed to be.
    cost_penalty = np.exp(-2.0 * (cost - 1.5) ** 2)
    
    return ei * cost_penalty
# EVOLVE-BLOCK-END
