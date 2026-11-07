"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI).
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We add a small epsilon to sigma to prevent division by zero if sigma is exactly 0,
    # though in practice GP sigma is rarely exactly 0 for new points.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    scores = norm.cdf(z)
    
    return scores
# EVOLVE-BLOCK-END
