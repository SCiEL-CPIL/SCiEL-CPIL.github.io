import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (np.array): Posterior mean, shape (N,)
    sigma (np.array): Posterior standard deviation, shape (N,)
    f_best (float): Best observed value so far.
    
    Returns:
    np.array: Scores for each candidate, shape (N,). Higher score means higher priority.
    """
    # Exploration parameter
    beta = 2.5
    
    # UCB for minimization:
    # We want to minimize the function.
    # The "optimistic" value for a point is mu - beta * sigma.
    # We want to pick the point with the lowest optimistic value.
    # Since the function returns scores where higher is better,
    # we return the negative of the optimistic value: - (mu - beta * sigma) = -mu + beta * sigma.
    scores = -mu + beta * sigma
    
    return scores
