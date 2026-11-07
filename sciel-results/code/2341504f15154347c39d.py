"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x).
    # The acquisition function should be high for candidates likely to be low.
    # LCB(x) = mu(x) - beta * sigma(x)
    # To maximize this score (higher is better), we return -LCB(x) or equivalent.
    # Actually, standard BO libraries often return the value to minimize, 
    # but here "higher meaning evaluate next".
    # So we want to return a score where higher is better.
    # Score = - (mu - beta * sigma) = -mu + beta * sigma
    
    beta = 2.0
    
    # Ensure sigma is non-negative and stable
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    return scores
# EVOLVE-BLOCK-END
