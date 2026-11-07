"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Adaptive UCB (LCB for minimization)
    # kappa decreases as f_best improves (approaches 0)
    # f_best is the current minimum observed value.
    # For Rastrigin, f_best starts high and drops to 0.
    
    # Ensure f_best is a scalar float
    f_best_val = float(np.min(f_best)) if hasattr(f_best, 'min') else float(f_best)
    
    # Adaptive kappa
    # If f_best is large, we want high exploration (high kappa).
    # If f_best is small, we want low exploration (low kappa).
    # Using an exponential decay with a floor.
    kappa = 4.0 * np.exp(-f_best_val / 20.0) + 0.5
    
    # LCB = mu - kappa * sigma
    # We want to maximize (f_best - LCB) / cost
    # Score = (f_best - (mu - kappa * sigma)) / cost
    # Score = (f_best - mu + kappa * sigma) / cost
    
    numerator = f_best_val - mu + kappa * sigma
    denominator = cost + 1e-9  # Avoid division by zero, though cost >= 1
    
    scores = numerator / denominator
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
