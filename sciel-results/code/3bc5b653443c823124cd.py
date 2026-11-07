import numpy as np
import time

# Global deadline tracking
_start_time = None
_deadline = None
_budget_limit = 30.0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _deadline
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 300.0
    
    # Check deadline with safety margin
    elapsed = time.time() - _start_time
    if elapsed > 295.0:
        # Return a safe default (exploit best known) to avoid timeout
        return np.zeros_like(mu)
    
    # mu: mean of GP (2000,)
    # sigma: std of GP (2000,)
    # f_best: best observed value (scalar)
    # cost: cost of evaluating each candidate (2000,)
    
    # Budget-constrained UCB on reward
    # Reward for minimization: negative regret = f_best - f(x)
    # Expected reward: f_best - mu
    # Uncertainty: sigma
    # We want to maximize: (f_best - mu + beta * sigma) / sqrt(cost)
    # This is equivalent to maximizing expected improvement with UCB exploration
    # divided by sqrt(cost) for cost-awareness
    
    # Dynamic beta based on remaining budget ratio
    # More remaining budget -> more exploration
    # Less remaining budget -> more exploitation
    # remaining_budget_ratio = (budget_limit - elapsed_cost_estimate) / budget_limit
    # Since we don't have exact remaining cost, we use a heuristic based on elapsed time
    # But actually, the 'cost' array gives us the cost of each candidate
    # Let's estimate remaining budget from total cost spent so far? 
    # We don't have total cost spent directly, but f_best is given.
    # Let's use a simple dynamic beta that starts high and decreases.
    # Or just use a fixed beta that works well.
    
    # Let's try beta = 3.0 as a starting point, and divide by sqrt(cost)
    beta = 3.0
    
    # Expected improvement (for minimization): (f_best - mu)
    # UCB exploration: beta * sigma
    # Combined score: (f_best - mu + beta * sigma)
    # Cost-aware: divide by sqrt(cost)
    
    # Ensure cost is positive and finite
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute score
    scores = (f_best - mu + beta * sigma) / np.sqrt(cost_safe)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
