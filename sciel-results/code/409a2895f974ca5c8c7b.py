"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses a hybrid of Expected Improvement and Lower Confidence Bound with
    adaptive exploration based on the number of evaluations inferred from sigma.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    # Use standard normal CDF and PDF
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Lower Confidence Bound (LCB) with adaptive beta
    # Early on, we want more exploration; later, more exploitation
    # Use the mean sigma as a proxy for exploration level
    mean_sigma = np.mean(sigma_safe)
    # Beta decreases as mean sigma decreases (more confident)
    # In early iterations, sigma is high -> beta should be high
    # In later iterations, sigma is low -> beta should be low
    beta = 0.8 + 0.4 * np.exp(-5 * mean_sigma)
    
    lcb = mu - beta * sigma_safe
    
    # Normalize both EI and LCB to comparable scales
    ei_max = np.max(ei) if np.any(ei > 0) else 1.0
    lcb_min = np.min(lcb)
    lcb_range = np.max(lcb) - lcb_min + 1e-8
    
    # Blend EI and LCB
    # Use a weight that shifts from LCB to EI as we get more confident
    # Higher mean_sigma means less confidence -> more LCB weight
    lcb_weight = 0.5 * (1 + np.tanh(2 * mean_sigma))
    ei_weight = 1 - lcb_weight
    
    # Normalize LCB to [0, 1] range for blending
    lcb_norm = (lcb - lcb_min) / lcb_range
    # Scale EI to similar range
    ei_norm = ei / (ei_max + 1e-8)
    
    # Combined score: higher is better
    score = ei_weight * ei_norm + lcb_weight * lcb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
