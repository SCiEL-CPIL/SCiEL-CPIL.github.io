import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 310.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses Probability of Improvement (PI) with adaptive exploration.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points, shape (N,).
    sigma (np.ndarray): Posterior standard deviation at candidate points, shape (N,).
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point, shape (N,). Higher score is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Fast deterministic fallback: simple LCB
        sigma_safe = np.maximum(sigma, 1e-9)
        return -(mu - 0.5 * sigma_safe)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Probability of Improvement for minimization
    # P(f(x) < f_best) = P(mu(x) + sigma(x)*Z < f_best) where Z ~ N(0,1)
    # = P(Z < (f_best - mu(x)) / sigma(x))
    # = Phi((f_best - mu(x)) / sigma(x))
    z = (f_best - mu) / sigma
    pi_score = norm.cdf(z)
    
    # Add a small exploration term to encourage visiting uncertain regions
    # when the current best is very good (close to minimum)
    exploration_weight = 0.1
    sigma_normalized = sigma / (np.max(sigma) + 1e-9)
    scores = pi_score + exploration_weight * sigma_normalized
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
