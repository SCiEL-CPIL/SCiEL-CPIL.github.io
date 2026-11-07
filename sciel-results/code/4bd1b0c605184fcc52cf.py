"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement and UCB.
    Returns scores where higher values indicate more promising candidates.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Upper Confidence Bound (UCB) for minimization: we want low values,
    # so we use a "lower confidence bound" in spirit: mu - kappa * sigma
    # But for scoring where higher = better, we negate or transform.
    # Actually, for minimization with UCB-style exploration:
    # We want to explore where sigma is high AND mu is low.
    # A common approach: score = -mu + kappa * sigma (higher is better)
    kappa = 1.5
    ucb_score = -mu + kappa * sigma_safe
    
    # Normalize both to [0, 1] range for fair combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.ones_like(ei) * 0.5
    
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb_score) * 0.5
    
    # Weighted combination: 60% EI, 40% UCB
    weights_ei = 0.6
    weights_ucb = 0.4
    scores = weights_ei * ei_norm + weights_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
