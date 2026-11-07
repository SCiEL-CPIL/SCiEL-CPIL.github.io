import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Posterior Sampling of Regret (PSR) / Thompson Sampling acquisition.
    Samples from the posterior at each candidate point and returns the
    negative sampled value (so that higher score = lower expected value =
    more promising for minimization).
    
    For minimization, we want to find points where the function value is
    low. We sample y_i ~ N(mu_i, sigma_i^2) for each candidate i, and
    return -y_i as the score (higher means lower sampled value, i.e.,
    more likely to be a new minimum).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Sample from the posterior at each candidate point
    # y_i ~ N(mu_i, sigma_i^2)
    # We use a single sample per candidate (Thompson Sampling style)
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # For minimization, we want to select the point with the lowest sampled value.
    # Since the evaluator picks the point with the HIGHEST acquisition score,
    # we return negative of the sample so that lower sampled values get higher scores.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
