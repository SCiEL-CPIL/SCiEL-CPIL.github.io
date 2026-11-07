"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard constant (seconds)
DEADLINE_LIMIT = 290.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses a Hybrid Expected Improvement with a square-root exploration bonus.
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division and sqrt issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure f_best is finite
    if not np.isfinite(f_best):
        # If f_best is not finite, we can't compute EI properly.
        # Return exploration-based score to keep moving.
        # This case is unlikely given the problem statement.
        scores = sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    # Standard Expected Improvement for minimization
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Exploration bonus: square root of sigma for smoother exploration
    # Coefficient 0.5 balances exploration and exploitation for this budget
    exploration_bonus = 0.5 * np.sqrt(sigma_safe)
    
    # Hybrid score
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_LIMIT:
        # Return current best effort
        pass
        
    return scores
# EVOLVE-BLOCK-END
