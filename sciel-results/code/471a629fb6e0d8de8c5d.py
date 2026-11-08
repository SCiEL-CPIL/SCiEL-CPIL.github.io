import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # UCB with beta = 1.0
    # Lower mu is better, so we want high (mu - beta*sigma) ? 
    # Wait, we are minimizing.
    # UCB for minimization: mu - beta * sigma
    # We want to MAXIMIZE the acquisition score.
    # So we should use -mu + beta * sigma ?
    # Or just maximize (mu - beta * sigma) ? No, that favors low mu.
    # Standard UCB for minimization: q(x) = mu(x) - beta * sigma(x)
    # We want to minimize q(x).
    # So we should return -q(x) to maximize?
    # Or return (beta * sigma - mu) ?
    # Let's check: if mu is low (good), -mu is high. If sigma is high (uncertain), beta*sigma is high.
    # So score = beta * sigma - mu.
    # Let's divide by cost.
    
    beta = 1.0
    scores = (beta * sigma - mu) / (cost + 1e-9)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
