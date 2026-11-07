import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # UCB-style acquisition with cost normalization
    # Add small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploration parameter: start high, but since we don't know iteration,
    # use a fixed reasonable value. Beta ~ 2-4 is typical for UCB.
    beta = 2.0
    
    # UCB score: we want to minimize, so lower mu is better, higher sigma is more exploratory
    # score = (f_best - mu - beta * sigma) / cost
    # But this can be negative if mu > f_best, which is fine (we prefer positive scores)
    
    ucb_score = (f_best - mu - beta * sigma_safe) / cost
    
    # Also compute EI for blending
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_score = ei / cost
    
    # Blend: use the maximum to be more robust
    score = np.maximum(ucb_score, ei_score)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
