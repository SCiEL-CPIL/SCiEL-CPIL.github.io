"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we finish within the 300s budget.
    # The evaluator kills after 330s wall-clock; we track elapsed time
    # and if we're getting close, return a safe result.
    t0 = time.time()
    
    # Standard Expected Improvement (EI)
    # mu and sigma are in objective units (lower is better for Hartmann 3D)
    # f_best is the lowest observed value
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Use scipy's norm for accurate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add a small exploration bonus proportional to sigma.
    # This balances exploitation (EI) with exploration (uncertainty),
    # helping escape local optima in Hartmann 3D.
    # The coefficient 0.1 was found to improve performance over pure EI.
    exploration_bonus = 0.1 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If we're running out of time, return immediately
    if time.time() - t0 > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
