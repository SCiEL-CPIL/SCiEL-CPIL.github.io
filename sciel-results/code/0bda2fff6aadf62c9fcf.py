"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Global variable to track the number of acquisition calls for adaptive beta
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Total evaluations planned is 30.
    # We want to explore heavily at the start and exploit at the end.
    # Let's use a linear decay of beta from 2.0 to 0.5 over the 30 steps.
    # Step 1: beta=2.0, Step 30: beta=0.5
    # However, the first 5 points are random, so the GP is very poor initially.
    # The acquisition is called for steps 6 through 30 (25 calls).
    # But _call_count starts at 1 for the first acquisition call (step 6).
    
    # Let's map call_count to a progress fraction.
    # If we assume 25 calls total for the BO loop:
    # progress = min(1.0, (_call_count - 1) / 24.0)
    
    # Actually, a simpler and robust approach for ST1D is often a fixed UCB with a moderate beta,
    # or a beta that depends on the uncertainty relative to the best.
    
    # Let's try an adaptive beta based on the call count, assuming ~25 BO steps.
    # If _call_count is 1, we are at the first BO step.
    # If _call_count is 25, we are at the last BO step.
    
    # Clamp call count to reasonable range
    c = _call_count
    if c < 1:
        c = 1
    if c > 30:
        c = 30
        
    # Linear interpolation of beta from 2.0 (exploration) to 0.5 (exploitation)
    # We want high exploration early to escape local minima.
    beta = 2.0 - (2.0 - 0.5) * ((c - 1) / 24.0)
    
    # Ensure beta is positive
    beta = max(beta, 0.1)
    
    # UCB score for minimization:
    # We want to select the point with the lowest expected value, but also high uncertainty.
    # score = mu - beta * sigma
    # Higher score means more promising (lower mu, higher sigma).
    
    # Handle numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    scores = mu - beta * sigma_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
