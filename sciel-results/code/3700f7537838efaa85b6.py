import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    This function is designed to minimize the objective function.
    It returns scores where higher values indicate a higher priority for evaluation.
    
    Parameters:
    mu (numpy array): Mean of the GP posterior.
    sigma (numpy array): Standard deviation of the GP posterior.
    f_best (float): The best observed value so far.
    
    Returns:
    numpy array: Scores for each candidate.
    """
    # Ensure sigma is strictly positive to avoid division by zero or negative variance issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Exploration parameter beta.
    # A value of 2.5 is chosen to balance exploration and exploitation
    # for the small evaluation budget (30 points) of the Branin function.
    beta = 2.5
    
    # UCB for minimization:
    # We are optimistic that the true value is mu - beta * sigma.
    # We want to minimize this value.
    # Therefore, the score to maximize is the negative of this optimistic value.
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
