"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) acquisition function.
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    # This directly maximizes the probability of finding a better point,
    # offering aggressive exploitation while still accounting for uncertainty.
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized distance from current best
    z = (f_best - mu) / sigma_safe
    
    # Probability that f(x) < f_best under the GP posterior
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
