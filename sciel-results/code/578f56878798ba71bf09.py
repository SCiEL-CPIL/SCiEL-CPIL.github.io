"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling: draw a single sample from the GP posterior
    # and select the candidate that minimizes (sample / cost).
    # Since we want to return scores where higher = evaluate next,
    # we return - (sample / cost).
    
    # Draw a single random standard normal sample
    z = np.random.standard_normal()
    
    # GP posterior sample: mu - sigma * z (minimization, so we subtract)
    # Actually, for minimization, the posterior sample is mu + sigma * z,
    # but since we want to find the minimum, we minimize the sample.
    # The sample from the posterior is: sample = mu + sigma * z
    # We want to minimize sample / cost, so score = - (sample / cost)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Draw the sample
    sample = mu + sigma * z
    
    # Compute score: higher is better, so we negate the ratio
    # Avoid division by zero or very small costs
    cost_safe = np.maximum(cost, 1e-9)
    
    score = -sample / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
