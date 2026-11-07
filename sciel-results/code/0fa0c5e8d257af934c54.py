import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using Thompson Sampling with cost-awareness.
    
    Strategy:
    1. Draw a single sample from the GP posterior: y ~ N(mu, sigma^2).
    2. Compute a cost-adjusted score: score = -y / cost.
       This favors points with low predicted function values and low costs.
    3. Return the scores. Higher scores are preferred.
    
    Deadline Guard:
    If we are running out of time, fall back to a fast deterministic heuristic (-mu / cost)
    to ensure valid output is returned.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    # This is much faster than generating random numbers and doing divisions? 
    # Actually, -mu/cost is about the same speed. 
    # But if we are in a tight loop, maybe just return -mu to be safe?
    # The prompt says "return the best result found with a safety margin".
    # If we are close to deadline, we should return a valid array quickly.
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma and cost
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    # Thompson Sampling: Draw a sample from the posterior
    # y ~ N(mu, sigma^2)
    # We generate standard normal noise z ~ N(0, 1)
    # y = mu + sigma * z
    # We want to minimize f, so we look for low y.
    # The score should be high for low y.
    # score = -y / cost
    
    z = np.random.normal(0, 1, size=mu.shape)
    y_sample = mu + sigma * z
    
    # Calculate cost-aware score
    # We want to maximize this score.
    # Low y_sample is good. Low cost is good.
    # So -y_sample / cost is a good proxy.
    scores = -y_sample / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
