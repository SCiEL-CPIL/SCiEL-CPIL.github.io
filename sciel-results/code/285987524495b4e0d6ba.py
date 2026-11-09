"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: sample a single function value per candidate
    from the Gaussian posterior and return those samples as scores.
    Higher sampled value means more likely to be the next evaluation target
    when the evaluator selects the candidate with the highest acquisition score
    for minimization (we negate so that lower true f gives higher score).
    
    Since we are minimizing, we want candidates with low predicted f.
    Thompson sampling: sample f_i ~ N(mu_i, sigma_i^2)
    We return -sampled_f so that the highest score corresponds to the lowest sampled f.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(size=mu.shape)
    
    # For minimization, we want to pick the candidate with the lowest sampled value.
    # The evaluator picks the candidate with the HIGHEST acquisition score.
    # So we return negative of the sampled values.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
