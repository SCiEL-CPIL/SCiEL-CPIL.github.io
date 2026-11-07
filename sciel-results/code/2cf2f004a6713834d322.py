"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Combines expected improvement with a smooth cost penalty to balance
    exploration of promising regions against evaluation cost.
    
    The cost-aware score is: EI / (1 + beta * cost_normalized)
    where cost_normalized scales the raw cost to [0, 1].
    
    This formulation:
    - Rewards candidates with high expected improvement
    - Penalizes expensive evaluations smoothly
    - Maintains numerical stability via safe sigma handling
    """
    # Start timer for deadline guard
    t_start = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for minimization
    # Improvement = f_best - mu (positive when mu < f_best)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Compute EI components with numerical stability
    # For large positive z: Phi(z) -> 1, phi(z) -> 0
    # For large negative z: Phi(z) -> 0, phi(z) -> 0
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = diff * cdf_z + sigma * pdf_z
    
    # Ensure EI is finite and non-negative
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment
    # cost is in [1, 2] (1 + mean(x) where x in [0,1])
    # Normalize cost to [0, 1] for smoother penalty
    cost_norm = (cost - 1.0) / 1.0  # Now in [0, 1]
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    
    # Smooth cost penalty: divide by (1 + beta * cost_norm)
    # beta controls the strength of cost awareness
    # beta=0 gives pure EI; higher beta penalizes expensive points more
    beta = 0.5
    cost_penalty = 1.0 + beta * cost_norm
    
    # Cost-aware acquisition score
    score = ei / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard: if we're getting close to timeout, return current result
    # The computation is vectorized so it should be fast, but add safety margin
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
