"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()


def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using Expected Improvement with adaptive beta.
    Uses a simple, robust EI-based approach that balances exploration and exploitation.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero and ensure positive values
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement
    # For minimization: improvement is f_best - mu(x) when mu(x) < f_best
    z = (f_best - mu) / sigma
    
    # Standard normal CDF and PDF using scipy
    from scipy.stats import norm
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI should be non-negative; clip any numerical issues
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus based on uncertainty
    # This encourages the algorithm to explore regions with high uncertainty
    # when the current best is still far from the optimum
    if f_best > 1.0:
        # Early stage: add more exploration
        exploration_bonus = 0.1 * sigma
    elif f_best > 0.1:
        # Middle stage: moderate exploration
        exploration_bonus = 0.05 * sigma
    else:
        # Late stage: mostly exploitation
        exploration_bonus = 0.01 * sigma
    
    score = ei + exploration_bonus
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
