"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Draws a sample from the posterior distribution N(mu, sigma^2) for each candidate.
    The score is the negative of the sampled value divided by the cost.
    Higher scores indicate candidates that are likely to have a lower objective value
    relative to their evaluation cost.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative costs (though cost should be positive)
    cost = np.maximum(cost, 1e-6)
    
    # Draw samples from the posterior
    # Using a single random seed per call is not possible as we don't control the evaluator's RNG,
    # but we can generate random noise here.
    n_samples = mu.shape[0]
    z = np.random.normal(0, 1, size=n_samples)
    
    # Sampled objective values
    y_samples = mu + sigma * z
    
    # We want to minimize y_samples.
    # The acquisition score should be high for good candidates.
    # Good candidates have low y_samples.
    # So we use -y_samples.
    # We also divide by cost to account for the cost-awareness.
    # If cost is higher, the score is lower for the same y_sample,
    # making cheaper points more attractive if their potential is similar.
    
    scores = -y_samples / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
