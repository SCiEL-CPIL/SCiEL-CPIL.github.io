import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) for minimization.
    
    For minimization, we seek points where the optimistic lower bound
    (mu - beta * sigma) is low.
    We convert this to a maximization score by measuring the potential
    improvement over f_best, adjusted by the exploration term,
    normalized by cost.
    
    Score = (f_best - (mu - beta * sigma)) / cost
          = (f_best - mu + beta * sigma) / cost
    """
    beta = 1.5
    
    # Calculate the optimistic value: mu - beta * sigma
    # We want this to be as small as possible.
    # So we want to maximize: -(mu - beta * sigma) = -mu + beta * sigma
    # To keep scores positive and comparable to EI, we shift by f_best:
    # f_best - mu + beta * sigma
    
    scores = (f_best - mu + beta * sigma) / (cost + 1e-9)
    
    # Replace any non-finite values with 0 to ensure validity
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
