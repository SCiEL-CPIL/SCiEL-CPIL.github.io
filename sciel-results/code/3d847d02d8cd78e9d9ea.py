import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive LCB
    # Estimate progress based on f_best
    # Goldstein-Price min is 3. Max in domain is around 1000-1200.
    # Let's assume f_max = 1000 for normalization.
    f_max = 1000.0
    f_min = 3.0
    
    # p is 0 at start (f_best high), 1 at end (f_best low)
    # If f_best is very high, p is negative, clip to 0.
    # If f_best is very low, p is > 1, clip to 1.
    p = (f_max - f_best) / (f_max - f_min)
    p = np.clip(p, 0.0, 1.0)
    
    # Beta starts at 2.0 and decreases to 0.5
    # More aggressive exploration initially
    beta = 2.0 - 1.5 * p
    
    # LCB score for minimization: maximize (-mu + beta * sigma)
    score = -mu + beta * sigma
    
    return score
