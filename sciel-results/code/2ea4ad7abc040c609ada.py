"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    This function implements Thompson Sampling by sampling from the posterior
    distribution of the objective function at each candidate point. The point
    with the lowest sampled value is selected. Since the evaluator picks the
    point with the highest acquisition score, we return the negative of the
    sampled values.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure sigma is non-negative and safe for operations
    # sigma is standard deviation, so it should be >= 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate standard normal random variables for all 2000 points
    # This is the 'randomness' in Thompson Sampling
    z = np.random.randn(len(mu))
    
    # Sample from the posterior distribution N(mu, sigma^2)
    # samples[i] ~ N(mu[i], sigma[i]^2)
    samples = mu + sigma_safe * z
    
    # The goal is minimization. We want to select the point with the lowest value.
    # The evaluator selects the point with the HIGHEST acquisition score.
    # Therefore, we return the negative of the sampled values.
    scores = -samples
    
    # Ensure all scores are finite to avoid invalid results
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
