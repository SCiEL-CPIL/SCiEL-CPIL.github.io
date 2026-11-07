"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is non-zero to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Use Probability of Improvement (PI)
    # PI = Phi((f_best - mu) / sigma)
    # This provides a bounded [0,1] score that balances exploration and exploitation
    return norm.cdf(z)
# EVOLVE-BLOCK-END
