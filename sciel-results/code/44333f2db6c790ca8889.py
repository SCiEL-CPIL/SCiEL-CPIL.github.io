# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    This approach uses the LCB metric, which balances exploitation (low mu) 
    and exploration (high sigma) linearly. This is distinct from Expected 
    Improvement, which integrates the distribution of improvements. LCB is 
    often more effective in multimodal landscapes like Ackley by maintaining 
    exploration pressure. The score is the negative LCB divided by cost, 
    prioritizing points with high potential for low function values per unit cost.
    
    Parameters:
    -----------
    mu : array-like
        Predicted mean of the GP at candidate points.
    sigma : array-like
        Predicted standard deviation of the GP at candidate points.
    f_best : float
        The best observed function value so far.
    cost : array-like
        The cost of evaluating each candidate point.
        
    Returns:
    --------
    scores : array-like
        Acquisition scores for each candidate point. Higher scores indicate 
        better candidates for the next evaluation.
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Calculate Lower Confidence Bound (LCB) for minimization
    # We want to minimize, so we look for low mu and high sigma (uncertainty)
    # LCB = mu - beta * sigma
    # We want to maximize the acquisition score, so we use -LCB
    # Score numerator = - (mu - beta * sigma) = beta * sigma - mu
    
    # Choose beta. A value around 2-3 is typical for exploration in BO.
    # We can make beta adaptive if needed, but a fixed value is a distinct design.
    beta = 2.5
    
    scores = beta * sigma_safe - mu
    
    # Cost-aware score: Score / cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = scores / cost_safe
    
    # Replace any non-finite values with 0 to ensure validity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
