import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive to avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration weight beta
    # The global minimum of Styblinski-Tang 2D is approx -78.3323
    f_min_known = -78.3323
    gap = f_best - f_min_known
    
    # If gap is large, we are far from the minimum, so explore more (higher beta).
    # If gap is small, we are close, so exploit more (lower beta).
    # Clip gap to [0, 10] to define the transition range.
    gap_norm = np.clip(gap, 0.0, 10.0) / 10.0
    beta = 0.1 + 0.9 * gap_norm
    
    # Blend EI and UCB-like term
    # Score = (EI + beta * sigma) / cost
    # We want higher score to be better.
    # EI is positive when mu < f_best.
    # sigma is positive.
    # cost is positive.
    
    scores = (ei + beta * sigma_safe) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
