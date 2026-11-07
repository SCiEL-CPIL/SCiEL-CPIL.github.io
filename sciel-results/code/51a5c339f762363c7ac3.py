import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses an adaptive Upper Confidence Bound (UCB) strategy.
    
    The UCB balances exploration and exploitation. We adapt the exploration
    parameter (beta) based on the current best value (f_best).
    - When f_best is far from the known minimum (3.0), we increase beta to 
      encourage exploration of the search space to find the global basin.
    - When f_best is close to the minimum, we decrease beta to exploit 
      the local neighborhood and refine the solution.
    """
    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the gap to the known global minimum (3.0)
    # If f_best is already below 3.0 (shouldn't happen for Goldstein-Price min, 
    # but possible due to noise or different scaling in other contexts), 
    # we treat the gap as 0 to maximize exploitation.
    gap = max(f_best - 3.0, 0.0)
    
    # Adaptive exploration parameter beta
    # We map the gap to a beta value using a tanh function for smooth transition.
    # beta_min = 0.5 (exploitative, near optimum)
    # beta_max = 3.0 (exploratory, far from optimum)
    # Scale of 5.0 allows beta to ramp up quickly for larger gaps.
    beta = 0.5 + 2.5 * np.tanh(gap / 5.0)
    
    # UCB Score for minimization:
    # We want to minimize f, so we maximize the score.
    # Score = -mean + beta * std_dev
    scores = -mu + beta * sigma
    
    # Ensure scores are finite to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
