import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    sigma = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Probability of improvement (higher = more likely to beat f_best)
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Expected improvement (for exploitation)
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    
    # Blend: 70% exploitation (EI), 30% exploration (PI-based)
    # Actually, let's use a different blend
    # Score = (ei + beta * sigma) / cost
    # This is UCB-like
    
    # Structural change: use a "confidence-weighted improvement"
    # score = ei / (sigma + eps) / cost  -- this is like a t-statistic
    # High when improvement is large relative to uncertainty
    
    # Let's try: score = (f_best - mu) / (sigma + eps) / cost
    # This rewards points that are confidently better
    
    confidence_improvement = (f_best - mu) / (sigma + 1e-8)
    
    # Add a small exploration bonus for very uncertain regions
    exploration_bonus = 0.1 * sigma / (sigma + np.median(sigma) + 1e-8)
    
    score = (confidence_improvement + exploration_bonus) / cost_safe
    
    score = np.where(np.isfinite(score), score, 0.0)
    return score
