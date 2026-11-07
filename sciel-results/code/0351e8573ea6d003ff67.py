"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.special import erfc


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of:
    1. Expected Improvement (EI) scaled by cost
    2. Uncertainty bonus (sigma) to encourage exploration
    3. A small penalty for high cost to favor cheaper points
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety: handle edge cases
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for expected improvement
    # z = (f_best - mu) / sigma
    # For minimization, we want points where mu is much less than f_best
    delta = f_best - mu  # Positive when mu < f_best (good)
    z = delta / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    # Use erfc for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = 0.5 * erfc(-z / np.sqrt(2.0))
    
    ei = delta * Phi_z + sigma_safe * phi_z
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: cost is 1 + mean(x_scaled), so range is [1, 2]
    # We want to penalize higher cost but not too aggressively
    cost_safe = np.maximum(cost, 1e-6)
    
    # Primary score: EI per unit cost
    score_ei = ei / cost_safe
    
    # Exploration bonus: prefer high uncertainty regions
    # This helps escape local optima
    # Weight sigma by a factor to balance exploitation vs exploration
    exploration_weight = 0.1
    score_explore = exploration_weight * sigma_safe / cost_safe
    
    # Combine: weighted sum of EI/cost and exploration/cost
    # The EI term dominates when there's clear improvement potential
    # The exploration term helps when EI is small but uncertainty is high
    score = score_ei + score_explore
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Small additional penalty for very high cost to further prefer cheaper points
    # This is a gentle nudge, not a hard constraint
    cost_penalty = 0.05 * (cost_safe - 1.0)  # Range [0, 0.05]
    score = score - cost_penalty
    
    # Final check for finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - t0 > 280:
        # If we're running out of time, just return what we have
        pass
    
    return score
# EVOLVE-BLOCK-END
