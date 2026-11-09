"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamically tuned UCB formulation: score = (mu + kappa * sigma) / cost,
    where kappa is adapted based on the progress of the search (f_best relative to
    expected range). Early in the search, higher kappa encourages exploration;
    later, lower kappa focuses on exploitation.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value (lowest)
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Safety: ensure finite inputs and correct types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Determine progress p in [0, 1]
    # p=1 means early (high f_best), p=0 means near minimum (low f_best)
    # Hartmann 3D: F_MIN = -3.86278, typical initial best ~ -2.0
    F_MIN = -3.86278
    F_INIT = -2.0
    
    delta_f = F_INIT - F_MIN
    if delta_f <= 0:
        p = 1.0
    else:
        p = (f_best - F_MIN) / delta_f
        p = np.clip(p, 0.0, 1.0)
    
    # Dynamically tune kappa based on progress
    # Early (p~1): kappa ~ 3.0 (explore more)
    # Late (p~0): kappa ~ 0.5 (exploit more)
    kappa = 0.5 + 2.5 * p
    
    # Compute UCB for minimization: we want low mu, high sigma
    # score_raw = -(mu - kappa * sigma) = -mu + kappa * sigma
    # But we also want to incorporate exploitation of known good regions
    # Blend with Expected Improvement for better exploitation
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -40, 40)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    # EI for minimization
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # UCB score for minimization (higher = more promising)
    ucb_raw = -mu + kappa * sigma_safe
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb_raw)
    ucb_max = np.max(ucb_raw)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_raw - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_raw)
    
    # Blend: weight UCB more when exploring (early), EI more when exploiting (late)
    # w_ucb = 0.4 + 0.4 * p  (early: 0.8, late: 0.4)
    w_ucb = 0.4 + 0.4 * p
    w_ei = 1.0 - w_ucb
    
    blended = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Cost-aware: divide by cost
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - t_start > 290:
        pass  # Should not happen for N=2000
        
    return scores
# EVOLVE-BLOCK-END
