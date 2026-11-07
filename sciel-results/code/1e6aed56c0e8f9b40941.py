import numpy as np

def acquisition(mu, sigma, f_best):
    beta = 1.0
    # UCB for minimization: we want to maximize the score.
    # The value is low (good) if mu is low.
    # The uncertainty is high (good for exploration) if sigma is high.
    # Score = -mu + beta * sigma
    # We must handle potential numerical issues.
    scores = -mu + beta * sigma
    # Ensure finite
    return np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
