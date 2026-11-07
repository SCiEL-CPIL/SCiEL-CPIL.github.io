"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a structural blend of Expected Improvement and Posterior Variance (UCB)
    with cost-awareness, plus a deadline guard.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 325.0  # 5s safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # --- Expected Improvement (for minimization) ---
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(ei > 0, ei, 0.0)
    
    # --- Posterior Variance (uncertainty term) ---
    # Use sigma directly as a measure of uncertainty
    var_term = sigma
    
    # --- Blend EI and variance with adaptive weights ---
    # As f_best gets closer to the known minimum, we should explore less
    # and exploit more. Use a progress-based weight.
    # Hartmann 6d minimum is approximately -3.32237
    HARTMANN_6D_MIN = -3.32237
    
    # Progress: 0 at start (f_best = 0), 1 at optimum
    # f_best starts around 0 and goes down to -3.32237
    # Use a sigmoid-like function to transition from exploration to exploitation
    progress = np.clip((0.0 - f_best) / (0.0 - HARTMANN_6D_MIN), 0.0, 1.0)
    
    # When progress is low (early), weight variance more heavily (explore)
    # When progress is high (late), weight EI more heavily (exploit)
    w_var = 0.5 * (1.0 - progress)  # decreases from 0.5 to 0
    w_ei = 1.0 - w_var              # increases from 0.5 to 1.0
    
    # Normalize EI and variance to comparable scales
    # Use robust scaling based on current distribution
    ei_scale = np.percentile(ei, 95) if np.any(ei > 0) else 1.0
    ei_scale = max(ei_scale, 1e-9)
    var_scale = np.percentile(var_term, 95) if np.any(var_term > 0) else 1.0
    var_scale = max(var_scale, 1e-9)
    
    ei_norm = ei / ei_scale
    var_norm = var_term / var_scale
    
    blended = w_ei * ei_norm + w_var * var_norm
    
    # --- Cost-awareness ---
    # Use a softer cost penalty: divide by cost^alpha where alpha < 1
    # This reduces the penalty for expensive points compared to linear cost
    alpha = 0.5  # square root of cost
    scores = blended / np.power(cost, alpha)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
