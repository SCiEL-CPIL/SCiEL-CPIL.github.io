import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Structural change: Use Upper Confidence Bound (UCB) for minimization
    # instead of Expected Improvement (EI).
    # UCB explicitly balances exploitation (low mu) and exploration (high sigma).
    # This is often more robust in low-budget scenarios where EI might be
    # overly greedy or too conservative depending on f_best.
    
    # Beta controls the exploration-exploitation trade-off.
    # A value of 2.0 is a standard heuristic.
    beta = 2.0
    
    # Ensure sigma is positive to avoid division by zero in potential future uses
    # or numerical issues, though here we just multiply.
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization, the Lower Confidence Bound is L = mu - beta * sigma.
    # We want to minimize L, so we maximize -L = -mu + beta * sigma.
    ucb_score = -mu + beta * sigma
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations.
    # Cost is in [1, 2].
    scores = ucb_score / cost
    
    # Ensure output is finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
