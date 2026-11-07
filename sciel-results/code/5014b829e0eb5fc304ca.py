import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    mu (np.array): Mean of the Gaussian Process, shape (2000,).
    sigma (np.array): Standard deviation of the Gaussian Process, shape (2000,).
    f_best (float): Best observed value so far.
    cost (np.array): Cost of evaluating each candidate, shape (2000,).
    
    Returns:
    np.array: Acquisition scores, shape (2000,). Higher is better.
    """
    # UCB for minimization: We want to maximize -mu + beta * sigma
    # This encourages points with low mean (good) and high uncertainty (exploration)
    beta = 2.0
    
    # Calculate the UCB score
    ucb_scores = -mu + beta * sigma
    
    # Make it cost-aware by dividing by the cost
    # This penalizes expensive evaluations
    cost_safe = np.maximum(cost, 1e-9)
    scores = ucb_scores / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
