import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - kappa * sigma
    # Acquisition score is the "potential improvement" of LCB over f_best, divided by cost
    # Score = (f_best - (mu - kappa * sigma)) / cost
    
    # Clip sigma to avoid division by zero or extreme values, though GP noise is 1e-6
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Exploration parameter kappa
    # 1.645 corresponds to 95% confidence interval (one-sided)
    # Let's use 1.5 for a slightly more balanced exploration/exploitation
    kappa = 1.5
    
    lcb = mu - kappa * sigma_safe
    
    # We want to maximize the score.
    # If lcb < f_best, there is potential improvement.
    # Score = (f_best - lcb) / cost
    # If lcb > f_best, score is negative, which is fine, we just pick the max.
    
    scores = (f_best - lcb) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
