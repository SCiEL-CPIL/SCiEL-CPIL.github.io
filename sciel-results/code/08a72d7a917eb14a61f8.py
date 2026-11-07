import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Modified Expected Improvement with an exploration bonus
    # Score = EI(x) + lambda * sigma(x)
    # This stays within the "improvement" family but encourages exploration
    # similar to how KG balances exploitation and exploration.
    
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus
    # A small coefficient to encourage exploring high-uncertainty regions
    # without completely dominating the improvement signal.
    exploration_weight = 0.1
    bonus = exploration_weight * sigma_safe
    
    scores = ei + bonus
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
