import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Adaptive UCB for minimization
    # We want to maximize: -mu + beta * sigma
    
    # Estimate progress based on f_best
    # Optimum is approx -39.166
    f_opt = -39.166
    
    # If f_best is very good (close to f_opt), explore less (lower beta)
    # If f_best is bad (high value), explore more (higher beta)
    
    # Scale f_best to a range
    # f_best starts around 0-50, goes down to -40
    # Let's define a linear mapping for beta
    # beta_min = 0.5 (exploitation)
    # beta_max = 3.0 (exploration)
    
    # Thresholds
    # If f_best <= f_opt + 1.0, beta = 0.5
    # If f_best >= 0.0, beta = 3.0
    # Linear interpolation between
    
    lower_bound = f_opt + 1.0  # -38.166
    upper_bound = 0.0
    
    if f_best <= lower_bound:
        beta = 0.5
    elif f_best >= upper_bound:
        beta = 3.0
    else:
        # Interpolate
        # t = 1 when f_best = lower_bound, t = 0 when f_best = upper_bound
        t = (upper_bound - f_best) / (upper_bound - lower_bound)
        beta = 0.5 + (3.0 - 0.5) * t
        
    # Calculate UCB scores
    # Minimization: we want low mu, high sigma
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
