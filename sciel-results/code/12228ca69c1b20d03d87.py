"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    We draw a single sample from the posterior at each candidate point:
        theta = mu + sigma * Z, where Z ~ N(0, 1)
    
    Since we want to minimize the objective, we score candidates by -theta.
    Higher scores mean more likely to be the next best point to evaluate.
    
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
    
    # Thompson sample: theta = mu + sigma * Z
    theta = mu + sigma_safe * Z
    
    # Score: negative of the sample (since we minimize, lower theta is better)
    scores = -theta
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
