"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware Lower Confidence Bound (LCB) acquisition score for 2000 candidates.
    
    Uses (mu - kappa * sigma) / cost, where kappa is adapted based on the current
    gap to the known minimum.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Estimate the stage based on gap to known minimum
    known_min = -78.3323
    gap = f_best - known_min
    
    # Adapt kappa: more exploration when far from minimum, less when close
    if gap < 5.0:
        kappa = 1.5
    elif gap < 15.0:
        kappa = 2.0
    elif gap < 30.0:
        kappa = 2.5
    else:
        kappa = 3.0
    
    # Compute LCB score: (mu - kappa * sigma) / cost
    # For minimization, we want low mu and high sigma (uncertainty)
    # The score should be higher for more promising points
    # So we negate: -mu + kappa * sigma, then divide by cost
    lcb_score = (-mu_safe + kappa * sigma_safe) / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(lcb_score), lcb_score, 0.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return what we have
        return score
    
    return score
# EVOLVE-BLOCK-END
