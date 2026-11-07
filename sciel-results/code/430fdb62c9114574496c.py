"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 295.0  # safety margin before 330s kill
    return _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    deadline = _get_deadline()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical stability: avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Check deadline - if we're close, return a simple fallback
    if time.time() > deadline - 2.0:
        # Simple fallback: minimize sampled value / cost
        # Use deterministic approximation: (mu - sigma) / cost
        scores = (mu - sigma_safe) / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Thompson Sampling: draw a single sample from the GP posterior
    # f(x) ~ N(mu, sigma^2), so sample f_sample = mu - sigma * z where z ~ N(0,1)
    # We want to MINIMIZE f, so we sample and pick the minimum
    # Score should be HIGHER for better candidates, so we negate the sampled value
    # and divide by cost to account for evaluation cost
    
    # Draw a single standard normal sample for all candidates
    z = np.random.standard_normal(len(mu))
    
    # Sampled function value: f_sample = mu - sigma * z
    # (subtracting sigma*z gives a sample from N(mu, sigma^2))
    f_sample = mu - sigma_safe * z
    
    # We want to minimize f_sample, so the "score" for ranking (higher is better)
    # is -f_sample / cost, which rewards low function values and low costs
    score = -f_sample / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
