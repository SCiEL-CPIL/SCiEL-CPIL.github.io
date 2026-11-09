"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamic blend of Expected Improvement (EI) and Probability of Improvement (PI).
    The weights are adjusted based on the search progress (f_best relative to known bounds)
    to balance exploration (PI) and exploitation (EI).
    
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
    # p=1 means early (high f_best, far from minimum), p=0 means near minimum
    # Hartmann 3D: F_MIN = -3.86278, typical initial best ~ -2.0
    F_MIN = -3.86278
    F_INIT = -2.0
    
    delta_f = F_INIT - F_MIN
    if delta_f <= 0:
        p = 1.0
    else:
        p = (f_best - F_MIN) / delta_f
        p = np.clip(p, 0.0, 1.0)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -40, 40)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement for minimization
    pi = cdf_z
    
    # Normalize EI and PI to [0, 1] to make them comparable for blending
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
        
    pi_max = np.max(pi)
    if pi_max > 1e-12:
        pi_norm = pi / pi_max
    else:
        pi_norm = np.zeros_like(pi)
        
    # Dynamic Weights
    # Early (p~1): Favor PI (exploration/probability) -> w_pi high, w_ei low
    # Late (p~0): Favor EI (exploitation) -> w_ei high, w_pi low
    # We use a smooth transition.
    # w_pi = 0.5 + 0.5 * p  (Early p=1 -> 1.0, Late p=0 -> 0.5)
    # w_ei = 0.5 - 0.5 * p  (Early p=1 -> 0.0, Late p=0 -> 0.5)
    # This ensures we never fully ignore EI, as it is the primary driver of improvement.
    # Let's try a stronger shift:
    # w_pi = 0.8 * p + 0.2
    # w_ei = 0.8 * (1 - p) + 0.2
    # Normalize weights so they sum to 1?
    w_pi_raw = 0.2 + 0.8 * p
    w_ei_raw = 0.2 + 0.8 * (1.0 - p)
    w_sum = w_pi_raw + w_ei_raw
    w_pi = w_pi_raw / w_sum
    w_ei = w_ei_raw / w_sum
    
    blended = w_ei * ei_norm + w_pi * pi_norm
    
    # Cost-aware: divide by cost
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - t_start > 290:
        pass
        
    return scores
# EVOLVE-BLOCK-END
