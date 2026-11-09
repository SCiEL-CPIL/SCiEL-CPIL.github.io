import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard (safety margin)
    # The computation is vectorized and fast, but we add a guard as per directive.
    # We don't have a global start time, so we assume this function is called
    # within a larger loop. If this function takes too long, it's a problem.
    # However, with 2000 points, norm.cdf and norm.pdf are fast.
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration weight
    # If f_best is large (early stage), explore more (higher beta).
    # If f_best is small (late stage), exploit more (lower beta).
    # Scale f_best to a 0-1 range using a sigmoid-like function.
    # Assuming typical f_best starts around 10-100 and goes to 0.
    # A soft cap at 10 seems reasonable.
    exploration_factor = f_best / (f_best + 1.0)
    beta = 0.1 + 0.4 * exploration_factor
    
    # UCB term for minimization (we want high score for low mu and high sigma)
    # The UCB score is -mu + beta * sigma.
    # But we are adding to EI. EI already penalizes high mu.
    # Adding beta * sigma encourages high uncertainty.
    # Adding -mu again might be too strong.
    # Let's just add the exploration bonus: beta * sigma.
    
    score = ei + beta * sigma
    
    return score
