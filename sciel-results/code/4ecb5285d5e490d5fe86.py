import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is not zero to avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: Uncertainty (Sigma)
    # We add a term proportional to sigma to encourage exploration
    # The coefficient 0.5 balances exploitation (EI) and exploration (Sigma)
    exploration_bonus = 0.5 * sigma_safe
    
    # Total Score
    score = EI + exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
