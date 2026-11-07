import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB
    # Start with beta=2.0, decrease to 1.0 as f_best approaches 0
    beta = 1.0 + 1.0 * np.exp(-f_best / 10.0)
    scores = -mu + beta * sigma
    return scores
