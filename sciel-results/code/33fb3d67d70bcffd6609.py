"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Standard Thompson Sampling acquisition function.
    
    We draw a single sample from the posterior predictive distribution at each
    candidate point. The sample is given by theta = mu + sigma * Z, where Z is
    a standard normal random variable. We then select the candidate with the
    lowest sampled value (minimizing the objective).
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std in objective units
        f_best: float - lowest observed value so far
    
    Returns:
        scores: array of shape (2000,) - finite scores, higher = evaluate next
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw standard normal noise
    Z = np.random.standard_normal(len(mu))
    
    # Standard Thompson sample
    theta = mu + sigma_safe * Z
    
    # Score: negative of the sample (since we minimize, lower theta is better)
    # We return negative theta so that higher scores correspond to lower expected values
    scores = -theta
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
