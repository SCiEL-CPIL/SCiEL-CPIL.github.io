import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted UCB.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far.
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Structural Change: Use Cost-Adjusted UCB
    # UCB for minimization: -mu + beta * sigma
    # We use a fixed beta=2.0 to encourage strong exploration,
    # especially given the low number of initial points (5).
    # The cost penalty is sqrt(cost) to avoid over-penalizing
    # points that are slightly more expensive but potentially much better.
    
    beta = 2.0
    
    ucb_scores = -mu + beta * sigma_safe
    
    # Apply cost penalty: divide by sqrt(cost)
    # This rewards low cost but less aggressively than dividing by cost
    scores = ucb_scores / np.sqrt(cost_safe)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback (exploitative) if we are running out of time
        # Although this shouldn't happen for 2000 points with simple ops
        return -mu / np.sqrt(cost_safe)
    
    return scores
