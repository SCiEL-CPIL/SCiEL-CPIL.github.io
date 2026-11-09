# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Draw a single sample from the posterior predictive distribution
    # For minimization, we want to find the point with the lowest predicted value.
    # We sample f ~ N(mu, sigma) and return -f so that higher scores correspond to lower f.
    z = np.random.standard_normal(len(mu))
    samples = mu - sigma * z
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
