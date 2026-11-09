import numpy as np
from scipy.stats import norm
import time

_state = {
    "start_time": None
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma.
    This balances exploitation (low mu) and exploration (high sigma).
    """
    if _state["start_time"] is None:
        _state["start_time"] = time.time()
    
    elapsed = time.time() - _state["start_time"]
    if elapsed > 325.0:
        # Fallback to exploitation if near deadline
        mu = np.asarray(mu, dtype=np.float64)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # UCB for minimization: we want to minimize mu - beta * sigma
    # Since higher score is better for selection, we return - (mu - beta * sigma)
    # score = -mu + beta * sigma
    beta = 2.5
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
