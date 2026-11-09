"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement with exploration bonus.
    
    Combines EI (expected reduction in f_best) with a sigma-based exploration
    term, both normalized by evaluation cost.
    """
    # Track elapsed time to ensure we finish within the deadline
    t0 = time.time()
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Expected improvement (EI)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware EI
    acq = ei / cost_safe
    
    # Exploration term: normalized sigma to encourage visiting uncertain regions
    # This helps escape local optima in the acquisition landscape
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        exploration_bonus = (sigma_safe / sigma_max) * 0.05 / cost_safe
    else:
        exploration_bonus = 0.0
    
    acq = acq + exploration_bonus
    
    # Ensure all values are finite
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    # Check time - if we're running out, return immediately
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin before 300s limit
        return acq
    
    return acq
# EVOLVE-BLOCK-END
