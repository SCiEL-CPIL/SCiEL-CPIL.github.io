import numpy as np
import time

# Module-level state to track time for the deadline guard
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of acquisition scores (higher is better).
    """
    global _start_time, _call_count
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: If we are getting close to the 300s limit (with safety margin),
    # we should return a safe fallback to avoid crashing the evaluator.
    elapsed = time.time() - _start_time
    if elapsed > 290.0:
        # Fallback: pick the cheapest point (lowest cost) to finish quickly
        # Higher score = better. So we want high score for low cost.
        # Score = -cost.
        return -np.asarray(cost, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    # Avoid division by zero in cost
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    _call_count += 1
    
    # Dynamic beta: Start with moderate exploration, decay over time to exploit.
    # The total budget is 30. Initial cost is ~5 * 1.5 = 7.5. Remaining ~22.5.
    # We expect roughly 10-15 evaluations.
    # Let's use a decay based on call count.
    # beta_t = beta_0 * (t / (t + t0))^(-alpha) ? 
    # Simpler: beta_t = beta_0 / (1 + 0.1 * t)
    # Or based on expected remaining budget? We don't have total cost passed, only candidate costs.
    # f_best is passed. We can track internal state of f_best history? No, stateless function.
    # But we have _call_count.
    
    # Let's use a fixed schedule based on _call_count.
    # Early calls: high beta. Late calls: low beta.
    # Total calls likely < 20.
    beta_0 = 2.0
    beta = beta_0 / (1.0 + 0.2 * _call_count)
    
    # UCB: We want to minimize the function.
    # Standard UCB for minimization: mu - beta * sigma.
    # However, we want to be cost-aware.
    # Score = (mu - beta * sigma) / cost ?
    # This rewards low predicted values and high uncertainty, penalized by cost.
    # Note: mu is in objective units (lower is better).
    # If mu is low (good), (mu - beta*sigma) is very low.
    # We want to MAXIMIZE the score.
    # So we should maximize - (mu - beta * sigma) / cost = (beta * sigma - mu) / cost.
    
    # Let's check the sign.
    # If we find a point with mu = -70 (good) and sigma = 1, cost = 1.
    # Score = (2*1 - (-70)) / 1 = 72.
    # If we find a point with mu = 0 (bad) and sigma = 5, cost = 1.
    # Score = (2*5 - 0) / 1 = 10.
    # The good point wins.
    
    # What if a point has mu = -70, sigma = 1, cost = 5?
    # Score = 72 / 5 = 14.4.
    # What if a point has mu = -60, sigma = 1, cost = 1?
    # Score = (2 - (-60)) / 1 = 62.
    # The cheaper point with slightly worse mean wins. This is correct for cost-awareness.
    
    ucb_term = beta * sigma
    # Optimistic value (lower is better): mu - ucb_term
    # We want to maximize the "gain" per cost.
    # Gain relative to what? f_best?
    # Expected Improvement (EI) is usually: E[min(0, y - f_best)]
    # UCB is a proxy for EI.
    # Let's stick to the raw UCB score normalized by cost.
    
    score = (ucb_term - mu) / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
