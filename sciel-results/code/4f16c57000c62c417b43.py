import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for Bayesian optimization of Goldstein-Price 2D.
    Draws a single sample path from the posterior N(mu, sigma^2) and returns it as scores.
    This provides natural exploration-exploitation balance without tuning coefficients.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0
    
    # Thompson Sampling: draw one sample from the posterior
    # The evaluator picks the candidate with the highest score, so we want to
    # sample from the posterior in a way that favors lower function values.
    # We sample f(x) ~ N(mu(x), sigma(x)^2) and return -f_sample so that
    # higher scores correspond to lower predicted function values.
    
    # Use a single random draw for all candidates (one sample path)
    rng = np.random.RandomState()
    eps = rng.standard_normal(len(mu))
    
    # Sample from posterior: f = mu + sigma * eps
    sample = mu + sigma * eps
    
    # Return negative sample so that higher score = lower predicted value
    scores = -sample
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
