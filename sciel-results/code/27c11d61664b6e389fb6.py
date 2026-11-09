import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    Samples from the posterior predictive distribution and returns samples as scores.
    Higher scores indicate more promising candidates.
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: sample from N(mu, sigma^2)
    # For minimization, we want to find points with low f values
    # So we sample f ~ N(mu, sigma^2) and return -f as the score (higher is better)
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # For minimization, we want to maximize the negative of the sample
    # This way, points where the sampled function value is low get high scores
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 1.0:
        pass
        
    return scores
