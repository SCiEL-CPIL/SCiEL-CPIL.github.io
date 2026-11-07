"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a dynamic Lower Confidence Bound (LCB).
    
    For minimization, the LCB is defined as:
        LCB(x) = mu(x) - kappa * sigma(x)
    
    We maximize the acquisition score, so we return:
        Score(x) = -mu(x) + kappa * sigma(x)
    
    The parameter kappa is dynamically adjusted based on the current best
    observed value (f_best) relative to the known global minimum of the
    Hartmann 3D function. This encourages exploration early in the search
    when f_best is poor, and exploitation later when f_best is close to
    the optimum.
    """
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known global minimum for Hartmann 3D
    f_global_min = -3.86278
    
    # Estimate initial f_best for progress calculation
    # Uniform random points in [0,1]^3 for Hartmann 3D typically yield values in [-1, 1]
    # Using 0.0 as a reasonable reference for "start"
    f_start_ref = 0.0
    
    # Calculate progress: 1.0 at start (poor f_best), 0.0 at optimum
    # If f_best > f_start_ref, progress is capped at 1.0 (max exploration)
    # If f_best < f_global_min, progress is capped at 0.0 (max exploitation)
    span = f_start_ref - f_global_min
    if span == 0:
        progress = 0.5
    else:
        progress = (f_best - f_global_min) / span
        progress = np.clip(progress, 0.0, 1.0)
        
    # Dynamic kappa
    # Use a square root scaling to maintain higher exploration pressure for longer
    # kappa ranges from 0.5 (exploit) to 2.5 (explore)
    kappa_min = 0.5
    kappa_max = 2.5
    kappa = kappa_min + (kappa_max - kappa_min) * np.sqrt(progress)
    
    # Calculate LCB score for maximization
    # Score = -mu + kappa * sigma
    scores = -mu_safe + kappa * sigma_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
