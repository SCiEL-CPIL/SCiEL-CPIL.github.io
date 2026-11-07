import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive Beta
    # f_best is typically large initially (e.g., 10-100) and small later (e.g., 0-1)
    # We want more exploration (higher beta) when f_best is large.
    # Sigmoid mapping:
    # If f_best = 10, sigmoid(10-5)=sigmoid(5)~0.99 -> beta ~ 3.98
    # If f_best = 5, sigmoid(0)=0.5 -> beta ~ 3.0
    # If f_best = 0, sigmoid(-5)~0.01 -> beta ~ 2.02
    # This seems reasonable.
    beta = 2.0 + 2.0 * (1.0 / (1.0 + np.exp(-(f_best - 5.0))))
    
    # Combine UCB and EI
    # UCB term: -mu + beta * sigma
    # EI term: ei
    # We weight EI slightly lower to ensure exploration via UCB
    score = (-mu + beta * sigma) + 0.5 * ei
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
