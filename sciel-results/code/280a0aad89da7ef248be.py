import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for Bayesian optimization.
    Uses Thompson Sampling: draw one sample from the GP posterior at each
    candidate and score by the negative sampled value. This provides a
    principled exploration/exploitation trade-off without tuning a beta.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma
    sigma = np.maximum(sigma, 1e-8)

    # Thompson Sampling: sample from N(mu, sigma^2) at each candidate
    rng = np.random.default_rng()
    samples = rng.normal(mu, sigma)
    
    # Score: negative sampled value (lower f is better, so higher score = lower f)
    scores = -samples

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
