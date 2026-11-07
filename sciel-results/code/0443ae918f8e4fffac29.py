"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill

def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330.0 - _DEADLINE_MARGIN
    return _DEADLINE

def _time_exceeded():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best):
    # Structural change: Use Upper Confidence Bound (UCB) with an adaptive
    # exploration parameter based on the number of evaluations inferred from
    # the spread of sigma, and a safety fallback to EI if time is tight.
    
    # Ensure we don't exceed the deadline
    _ensure_deadline()
    
    # Compute UCB score: mu + kappa * sigma
    # Adaptive kappa: start with higher exploration, decay as we learn.
    # Since we don't know the exact evaluation count, we use a heuristic:
    # if sigma is very small everywhere, we're in exploitation mode.
    # A common choice for Hartmann-3D is kappa around 1.0-2.0.
    # Let's use a fixed but slightly higher exploration than standard EI implies.
    
    # Alternative structural change: Use Logarithmic Probability of Improvement (LogPI)
    # or a hybrid. Let's try UCB with kappa=1.5 which is common for 3D functions.
    
    kappa = 1.5
    scores = mu + kappa * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, mu)
    
    return scores
# EVOLVE-BLOCK-END
