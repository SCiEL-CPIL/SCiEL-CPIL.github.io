"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive UCB.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure sigma is non-negative and has a minimum floor
    sigma = np.maximum(sigma, 1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Adaptive UCB parameter
    # The budget is 30. We estimate how much budget we have used.
    # Since we don't have the exact history, we can approximate.
    # A common heuristic is to make beta decay or grow based on progress.
    # For Griewank, which has many local minima, exploration is important early on.
    # As we get closer to the minimum, exploitation becomes more important.
    
    # A simple adaptive beta:
    # beta = sqrt(2 * log(t)) where t is the number of evaluations.
    # But we don't have t directly. We can use f_best as a proxy for progress?
    # Or simply use a fixed but well-tuned beta, and handle cost differently.
    
    # The current program uses UCB / cost. This penalizes expensive points heavily.
    # A better approach might be to subtract a cost penalty from the UCB score,
    # or to use a different cost-aware acquisition function.
    
    # Let's try: Score = UCB - lambda * cost
    # This way, we still get the exploration bonus, but we penalize high cost.
    # The lambda parameter controls the trade-off.
    
    # Alternatively, we can use the expected improvement (EI) which is naturally
    # more focused on finding the minimum, and then adjust for cost.
    
    # Let's stick with UCB but use a more nuanced cost handling.
    # The cost is 1 + mean(x scaled to [0, 1]). So cost is in [1, 2].
    # Dividing by cost is a strong penalty. Subtraction is milder.
    
    # Let's try: Score = (mu + beta * sigma) / (1 + cost)
    # This is a softer penalty than dividing by cost directly.
    
    beta = 2.0
    ucb_scores = mu + beta * sigma
    
    # Cost-aware score
    scores = ucb_scores / (1 + cost)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return scores
# EVOLVE-BLOCK-END
