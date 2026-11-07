"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard to prevent timeout
_DEADLINE = 325.0
_t0 = time.time()

def acquisition(mu, sigma, f_best):
    """
    Compute UCB acquisition scores for Bayesian optimization of 1D sphere.
    
    Args:
        mu (np.ndarray): Mean of the GP posterior.
        sigma (np.ndarray): Std dev of the GP posterior.
        f_best (float): Best observed value so far.
        
    Returns:
        np.ndarray: Acquisition scores. Higher is better.
    """
    # Check time limit
    if time.time() - _t0 > _DEADLINE:
        # Fallback: return negative mean (greedy exploitation)
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # UCB for minimization: -mu + beta * sigma
    # We use a fixed beta. For 1D smooth functions, beta=1.0 is a common starting point.
    # A larger beta encourages exploration.
    beta = 1.5
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
