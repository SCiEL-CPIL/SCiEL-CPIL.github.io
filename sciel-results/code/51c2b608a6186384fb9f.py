import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses Thompson Sampling: draw a single sample from the posterior predictive
    distribution f(x) ~ N(mu, sigma) for each candidate, and use the negative
    of that sample as the score (since we want to minimize).
    
    This is a structural change from the previous EI+UCB hybrid. Thompson Sampling
    naturally balances exploration and exploitation by sampling from the full
    posterior, rather than using a deterministic score.
    
    For minimization, we want to pick the point with the lowest sampled value,
    so score = -sample, where sample = mu + sigma * z, z ~ N(0,1).
    """
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a single standard normal sample for each candidate
    # This is the core of Thompson Sampling: sample from the posterior
    z = np.random.standard_normal(size=mu.shape)
    
    # Sample from the posterior predictive: f(x) ~ N(mu, sigma^2)
    sample = mu + sigma * z
    
    # For minimization, higher score means we should evaluate next
    # So we negate the sample: lower sampled f(x) -> higher score
    scores = -sample
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
