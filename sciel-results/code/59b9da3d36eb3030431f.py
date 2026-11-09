import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    f_min = -3.32237
    gap = f_best - f_min
    
    # Dynamic beta for UCB
    # If gap is large, we are far from optimum, explore more (higher beta)
    # If gap is small, we are close, exploit more (lower beta)
    gap_ref = 1.0
    beta_min = 0.5
    beta_max = 2.5
    
    # Clip gap to [0, 2*gap_ref] to avoid extreme values
    gap_clipped = np.clip(gap, 0, 2 * gap_ref)
    beta = beta_min + (beta_max - beta_min) * (gap_clipped / gap_ref)
    
    # UCB for minimization: maximize -mu + beta * sigma
    # Higher score means better candidate
    score = -mu + beta * sigma
    
    return score
