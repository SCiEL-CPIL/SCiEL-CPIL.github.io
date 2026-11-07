# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling-based acquisition function.
    
    For each candidate, we sample a single latent function value from the GP
    predictive distribution: y_k = mu_k + sigma_k * z_k where z_k ~ N(0,1).
    We then select the candidate that maximizes the ratio of this sampled
    (negative) value to its evaluation cost. Since we're minimizing, a lower
    sampled value is better, so we use -y_k / cost_k as the score.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviation predictions
        f_best: lowest observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Generate standard normal noise for Thompson Sampling
    # Use a fixed seed based on f_best to make it deterministic per call
    # but different across calls (since f_best changes as we find better points)
    rng = np.random.RandomState(int(abs(hash(f_best)) % (2**32)))
    z = rng.standard_normal(size=mu.shape)
    
    # Sample latent function values
    # y_k = mu_k + sigma_k * z_k
    y = mu + sigma * z
    
    # Since we're minimizing, lower y is better.
    # We want to maximize the score, so we use -y / cost
    # This gives higher scores to candidates with lower sampled values and lower costs
    scores = -y / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
