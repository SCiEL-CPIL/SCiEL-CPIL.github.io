import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate Expected Improvement (EI)
    # mu, sigma: shape (2000,)
    # f_best: scalar
    
    eps = 1e-9
    sigma_safe = sigma + eps
    
    z = (f_best - mu) / sigma_safe
    # EI formula for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus to encourage visiting uncertain regions
    # This helps in multimodal landscapes like Michalewicz
    exploration_weight = 0.5
    bonus = exploration_weight * sigma
    
    scores = ei + bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
