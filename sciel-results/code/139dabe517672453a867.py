"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a 2D RBF mixture.
    Uses a time-adaptive blend of Expected Improvement and a diversity-aware
    exploration term to balance exploitation and exploration.
    """
    t0 = time.time()
    deadline = 325.0  # safety margin before 330s kill

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Expected Improvement is a robust acquisition function that balances
    # exploration and exploitation without requiring ad-hoc normalization.
    # It is theoretically grounded and often outperforms heuristic blends.
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
