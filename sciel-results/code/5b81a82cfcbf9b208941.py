"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    This function uses a UCB strategy with a discrete, threshold-based 
    adaptation of the exploration parameter beta.
    
    - High f_best (early stage): High beta to aggressively explore 
      different basins in the multimodal Branin landscape.
    - Low f_best (late stage): Low beta to exploit the current best region.
    
    This structural change from POI (current) to UCB (inspiration) with 
    robust thresholding (refinement of inspiration 1) aims to balance 
    exploration and exploitation more effectively than fixed-beta UCB 
    or poorly-scaled adaptive UCB.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
        
    # Ensure sigma is non-negative and avoid division by zero later if needed
    sigma = np.maximum(sigma, 1e-10)
    
    # Known global minimum of Branin
    f_min = 0.397887
    
    # Adaptive Beta based on f_best thresholds
    # This avoids the potential scaling issues of linear interpolation 
    # seen in some adaptive implementations.
    if f_best > 5.0:
        beta = 3.0
    elif f_best > 1.5:
        beta = 1.5
    else:
        beta = 0.5
        
    # UCB score for minimization: maximize -mu + beta * sigma
    # We return the score; higher score means more likely to be selected.
    scores = -mu + beta * sigma
    
    # Handle any remaining non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
