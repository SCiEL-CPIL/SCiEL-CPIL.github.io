"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline checking
_START_TIME = time.time()
_DEADLINE = 300.0  # Safety margin from 330s limit

def _check_deadline():
    """Returns True if we are close to the deadline."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Upper Confidence Bound (UCB) for minimization (Lower Confidence Bound).
    Score = -mu + beta * sigma.
    Higher score means better candidate to evaluate next.
    """
    # Check deadline
    if _check_deadline():
        # Fallback: simple UCB or EI? Let's use simple UCB.
        beta = 2.5
        scores = -mu + beta * sigma
        # Ensure finite
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Standard UCB (LCB) for minimization
    # We want to minimize f(x).
    # GP: f(x) ~ N(mu, sigma^2)
    # We want to find x that minimizes the lower bound of f(x).
    # LCB = mu - beta * sigma
    # We maximize the score, so we maximize -LCB = -mu + beta * sigma.
    
    # Tuning beta:
    # beta = 2.5 is a common choice for balanced exploration/exploitation in 1D.
    beta = 2.5
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    # If sigma is 0, score is -mu.
    # If mu is inf, score is -inf.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
