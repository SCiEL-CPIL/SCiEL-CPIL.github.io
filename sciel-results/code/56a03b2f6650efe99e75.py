"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using LCB (Lower Confidence Bound) for minimization.
    
    Strategy:
    - Computes LCB = mu - beta * sigma
    - Higher score means more likely to be good (low value)
    - beta controls exploration vs exploitation
    - Includes a deadline guard to ensure the function returns within the time limit.
    """
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # Safety margin: if we are close to the deadline, return a fast greedy score
    if remaining_time < _DEADLINE_MARGIN:
        # Greedy: pick points with lowest mu (highest probability of being good)
        # Higher score is better, so we invert mu
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    # Use a moderate beta value for LCB
    # Lower beta means more exploitation, higher beta means more exploration
    beta = 1.5
    
    # LCB = mu - beta * sigma
    # For minimization, we want low values, so lower LCB is better
    # But we return higher scores as "better", so we negate LCB
    lcb = mu - beta * sigma
    scores = -lcb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
