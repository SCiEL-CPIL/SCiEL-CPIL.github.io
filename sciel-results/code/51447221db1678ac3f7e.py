import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # UCB term for minimization (maximize beta*sigma - mu)
    beta = 2.0
    ucb_term = beta * sigma_safe - mu
    
    # Adaptive weight for exploration (UCB) vs exploitation (EI)
    # w is high when f_best is high (early stage, need exploration)
    # w is low when f_best is low (late stage, need exploitation)
    # C is a scaling constant. 10 seems reasonable for Rastrigin.
    C = 10.0
    w = f_best / (f_best + C)
    
    # Combine
    # Note: ucb_term can be negative if mu is very large. 
    # ei is always non-negative.
    # We want to maximize the score.
    score = (w * ucb_term + (1 - w) * ei) / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
