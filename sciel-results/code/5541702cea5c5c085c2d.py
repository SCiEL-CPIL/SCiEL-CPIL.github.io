import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for minimization.
    
    Parameters:
    mu (np.ndarray): GP mean predictions (shape 2000,)
    sigma (np.ndarray): GP standard deviation predictions (shape 2000,)
    f_best (float): Best observed function value so far
    cost (np.ndarray): Cost of evaluating each candidate (shape 2000,)
    
    Returns:
    np.ndarray: Acquisition scores (shape 2000,), higher is better
    """
    # Exploration-exploitation trade-off parameter
    beta = 2.0
    
    # For minimization, we want to find points where the lower bound
    # (mu - beta*sigma) is significantly lower than f_best.
    # The "potential improvement" is f_best - (mu - beta*sigma)
    # = (f_best - mu) + beta*sigma
    
    potential_improvement = f_best - mu + beta * sigma
    
    # Cost-aware score: improvement per unit cost
    # cost is >= 1.0
    scores = potential_improvement / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
