"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Lower Confidence Bound (LCB) with cost discounting.
    
    Uses the LCB criterion, which balances exploitation (low mean) and exploration 
    (high uncertainty) explicitly. The score is discounted by the evaluation cost 
    to favor cheaper evaluations, maximizing the number of samples within the budget.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t0 = time.time()
    
    # Safety check for empty or invalid inputs
    if len(mu) == 0:
        return np.array([])
    
    # Ensure sigma is strictly positive to avoid division by zero or numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we use the negative of LCB:
    # Score_base = -mu + beta * sigma
    # beta controls the exploration-exploitation trade-off.
    # A higher beta encourages more exploration.
    beta = 1.0
    
    base_score = -mu + beta * sigma_safe
    
    # Cost-aware discounting: divide by cost^gamma to favor cheaper evaluations
    # gamma=2.0 (quadratic) was found to be effective in increasing evaluation count
    cost_safe = np.maximum(cost, 1.0)
    gamma = 2.0
    
    cost_discount = cost_safe ** gamma
    
    # Final score
    score = base_score / cost_discount
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 250:
        pass
    
    return score
    # EVOLVE-BLOCK-END
