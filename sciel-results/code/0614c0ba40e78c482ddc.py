import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) with cost awareness
    # For minimization, we maximize: -mu + beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # Beta controls the trade-off. 
    # A value of 1.0 is a standard choice for small budgets.
    beta = 1.0
    
    # Calculate UCB score
    ucb_scores = -mu + beta * sigma
    
    # Cost-aware scoring: higher score per cost is better
    # cost is given as 1 + mean(x_scaled), so it's at least 1.
    scores = ucb_scores / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
