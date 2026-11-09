import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) strategy for minimization.
    
    For minimization, we use the Lower Confidence Bound (LCB):
    LCB(x) = mu(x) - beta * sigma(x)
    We want to find x that minimizes LCB(x).
    Since the acquisition function must return scores where higher is better,
    we return -LCB(x) = -mu(x) + beta * sigma(x).
    
    Args:
        mu: Mean of the GP posterior at candidate points (shape: (2000,)).
        sigma: Standard deviation of the GP posterior at candidate points (shape: (2000,)).
        f_best: The lowest observed value so far.
        
    Returns:
        scores: Scores for each candidate point (shape: (2000,)). Higher is better.
    """
    # Hyperparameter for exploration vs exploitation
    # A value of 2.0 is a common starting point for UCB in BO
    beta = 2.0
    
    # Calculate scores: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite to satisfy the contract
    # Replace any non-finite values with a very low number so they are not selected
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
