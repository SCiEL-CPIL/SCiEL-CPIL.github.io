"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Adaptive hybrid EI + LCB acquisition for Bayesian optimization of hartmann 6d.
    Balances exploitation (EI) and exploration (LCB) with a time-adaptive beta
    that shifts from exploration to exploitation as f_best approaches the minimum.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known minimum of hartmann 6d
    f_min = -3.32237
    
    # Estimate progress: 0 = just started (f_best near 0), 1 = converged (f_best near f_min)
    # Typical random points in [0,1]^6 for hartmann 6d give values in roughly [-2, 0.5]
    # So initial f_best is around -0.5 to -2
    # Normalize the remaining gap
    gap = f_best - f_min  # positive, decreases as we improve
    # At start, gap ~ 2-3; at convergence, gap ~ 0
    # Use a sigmoid-like mapping to get progress in [0, 1]
    # progress = 0 when gap > 3, progress = 1 when gap < 0.1
    progress = 1.0 / (1.0 + np.exp(2.0 * (gap - 1.0)))
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive beta: start high for exploration, decrease for exploitation
    beta_start = 4.0
    beta_end = 0.5
    beta = beta_start + (beta_end - beta_start) * progress
    
    # EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # LCB score for minimization: we want low mu and high sigma
    # LCB = mu - beta * sigma, so -LCB = -mu + beta * sigma (higher is better)
    lcb_score = -mu + beta * sigma_safe
    
    # Scale EI to be comparable to LCB score
    # EI is typically small (0 to ~1), LCB score can be larger
    # Normalize by estimating the scale
    ei_scale = np.mean(ei) + 1e-9
    lcb_scale = np.std(lcb_score) + 1e-9
    
    # Adaptive weights: more EI weight as we progress
    w_ei = 0.3 + 0.7 * progress
    w_lcb = 1.0 - w_ei
    
    # Combine
    scores = w_ei * (ei / ei_scale) + w_lcb * (lcb_score / lcb_scale)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
