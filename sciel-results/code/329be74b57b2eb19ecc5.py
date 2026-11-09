"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time for the entire acquisition call
_ACQ_START_TIME = None
_DEADLINE = 330.0  # seconds, with safety margin
_SAFETY_MARGIN = 10.0  # seconds


def _check_deadline():
    """Check if we're approaching the deadline."""
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        return False
    elapsed = time.time() - _ACQ_START_TIME
    return elapsed > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses UCB (Upper Confidence Bound) for better exploration in multimodal landscapes.
    """
    global _ACQ_START_TIME
    _ACQ_START_TIME = time.time()
    
    # Check deadline early
    if _check_deadline():
        # Return a safe fallback: use -mu (pure exploitation)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # UCB acquisition: -mu + kappa * sigma
    # Higher scores mean more promising (lower mu, higher uncertainty)
    kappa = 1.0  # Exploration-exploitation balance
    
    scores = -mu + kappa * sigma
    
    # Ensure all scores are finite as required by the contract
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
