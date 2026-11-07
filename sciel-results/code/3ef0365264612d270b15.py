import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # f_best is the best (lowest) value observed so far
    # Ackley function minimum is 0.
    # We use f_best to estimate progress and adjust exploration (beta).
    
    # If f_best is high (> 2), we are in early stages -> high exploration.
    # If f_best is low (< 0), we are in late stages -> low exploration.
    # Linear interpolation of progress between f_best=2 and f_best=0.
    progress = (2.0 - f_best) / 2.0
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta controls the trade-off between exploitation (mu) and exploration (sigma).
    # Start with beta=3.0 (high exploration) and reduce to 0.5 (high exploitation).
    beta = 3.0 - 2.5 * progress
    
    # Upper Confidence Bound
    ucb = mu + beta * sigma
    
    # Cost-aware score: prefer points with high UCB and low cost
    scores = ucb / cost
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
