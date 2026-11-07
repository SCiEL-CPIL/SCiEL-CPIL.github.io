"""Cost-aware LCB acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time for the first call in a session
_lcb_start_time = None
_LCB_DEADLINE_S = 320.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    LCB balances exploitation (low mean) and exploration (high uncertainty)
    by selecting candidates with low lower confidence bounds. We use a
    cost-aware variant that penalizes expensive evaluations.
    
    The LCB score is: score = -(mu - beta * sigma) / cost^alpha
    where beta controls exploration and alpha controls cost sensitivity.
    """
    global _lcb_start_time
    
    # Initialize deadline tracking
    if _lcb_start_time is None:
        _lcb_start_time = time.time()
    
    # Check deadline
    elapsed = time.time() - _lcb_start_time
    if elapsed > _LCB_DEADLINE_S:
        # Return zeros to avoid biasing
        return np.zeros(len(mu))
    
    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-9)
    
    # LCB parameter: beta controls exploration-exploitation tradeoff
    # Start with moderate exploration, could adapt based on progress
    beta = 2.0
    
    # Compute LCB: lower confidence bound for minimization
    # We want to maximize the "improvement" score, so we negate the LCB
    lcb = mu - beta * s
    
    # Cost awareness: divide by a power of cost to penalize expensive evaluations
    c = np.maximum(cost, 1e-9)
    alpha = 0.7  # Cost sensitivity exponent
    cost_penalty = np.power(c, alpha)
    
    # Score: higher is better for next evaluation
    scores = -lcb / cost_penalty
    
    # Safety: Ensure finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
