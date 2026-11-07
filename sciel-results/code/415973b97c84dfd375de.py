"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Expected Improvement with adaptive exploration
    # Using a small epsilon to avoid division by zero
    eps = 1e-8
    safe_sigma = np.maximum(sigma, eps)
    
    # Standard EI
    z = (f_best - mu) / safe_sigma
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local minima in the RBF mixture
    # Scale exploration relative to current sigma
    exploration_weight = 0.05
    exploration_bonus = exploration_weight * safe_sigma
    
    # Combine EI with exploration
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
