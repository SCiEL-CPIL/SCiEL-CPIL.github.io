"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Pure GP-LCB (Lower Confidence Bound) for minimization
    # Score = -mu - kappa * sigma
    # Higher score means we prefer points with lower mu (better objective) 
    # or higher sigma (more uncertainty/exploration)
    
    # Safety check for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Fixed exploration parameter
    # A value around 2.0 is standard for LCB in 1D problems
    # This encourages exploration when uncertainty is high
    kappa = 2.0
    
    # Compute LCB score
    # We want to minimize f, so we look for low mu.
    # To encourage exploration, we add kappa * sigma.
    # The quantity to maximize is -mu + kappa * sigma? 
    # Wait. 
    # LCB for minimization: we want to find the point that minimizes the lower bound.
    # Lower bound of f(x) is mu(x) - kappa * sigma(x).
    # We want to select the x that minimizes this lower bound.
    # So the "value" to minimize is mu - kappa * sigma.
    # But the task says "Return 2000 finite scores, higher meaning evaluate next".
    # So we need to maximize a score.
    # Score = -(mu - kappa * sigma) = -mu + kappa * sigma.
    
    score = -mu + kappa * sigma_safe
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
