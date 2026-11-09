import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using UCB with adaptive beta.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta based on f_best
    # f_best starts around 25 (max of x^2 on [-5,5]) and goes to 0.
    # We want beta to start high (explore) and decrease (exploit).
    # beta = beta_min + (beta_max - beta_min) * (f_best / f_ref)
    # Let f_ref = 25.0
    # beta_min = 0.5, beta_max = 2.5
    # This ensures beta is always >= 0.5, avoiding pure greedy.
    
    f_ref = 25.0
    beta_min = 0.5
    beta_max = 2.5
    
    # Clip f_best to [0, f_ref] to handle cases where f_best might be slightly negative or > f_ref
    f_best_clipped = np.clip(f_best, 0.0, f_ref)
    
    # Linear interpolation
    beta = beta_min + (beta_max - beta_min) * (f_best_clipped / f_ref)
    
    # UCB score: -mu + sqrt(beta) * sigma
    # We want to maximize this score.
    score = -mu + np.sqrt(beta) * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
