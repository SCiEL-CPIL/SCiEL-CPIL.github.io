"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def _check_deadline():
    return time.time() < _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using
    Probability of Improvement (POI) normalized by cost.
    
    This treats the problem as a stochastic bandit where we maximize the
    probability of finding a point better than f_best, weighted by evaluation cost.
    POI = Phi((f_best - mu) / sigma)
    Score = POI / cost
    
    This is a structural change from Thompson Sampling: it uses a deterministic
    likelihood signal (probability of improvement) rather than a stochastic sample,
    and normalizes by cost to prefer cheaper evaluations when improvement probabilities
    are similar.
    """
    # Ensure finite inputs and proper types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline - if we're running out of time, return a safe default
    if not _check_deadline():
        # Return a simple score that favors low mu and low cost
        scores = -mu / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Compute improvement: f_best - mu
    # Where mu < f_best, improvement is positive (we expect to be better)
    improvement = f_best - mu
    
    # Compute z-score
    z = improvement / sigma
    
    # Probability of Improvement: P(y < f_best) = Phi(z)
    # For large positive z, POI -> 1; for large negative z, POI -> 0
    poi = norm.cdf(z)
    
    # Clip to avoid numerical issues (POI should be in [0, 1])
    poi = np.clip(poi, 1e-10, 1.0 - 1e-10)
    
    # Normalize by cost: prefer cheaper points when POI is similar
    scores = poi / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
