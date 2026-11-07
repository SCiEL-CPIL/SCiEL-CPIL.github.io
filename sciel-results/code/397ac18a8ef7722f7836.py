import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of improvement times sigma (exploration term)
    # This adds uncertainty-driven diversity
    p_improve = norm.cdf(z)
    exploration = p_improve * sigma
    
    # Combine EI and exploration term
    # EI provides principled exploitation, exploration adds diversity
    score = ei + 0.5 * exploration
    
    # Cost-aware penalty with sublinear discount
    # 1/sqrt(cost) allows exploration of higher-cost regions where global minimum may reside
    cost_discount = 1.0 / np.sqrt(cost)
    scores = score * cost_discount
    
    # Ensure outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
