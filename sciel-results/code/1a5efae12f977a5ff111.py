"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _branin2d(x1, x2):
    """Branin function for reference (not used in acquisition, but useful for sanity)."""
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (12.0 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement and UCB.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Upper Confidence Bound with adaptive beta
    # Use a beta that balances exploration and exploitation
    # When f_best is high (early), explore more; when low (late), exploit more
    # Normalize f_best relative to Branin's range [0.398, ~302]
    f_range = 302.0 - 0.398
    f_norm = np.clip((f_best - 0.398) / f_range, 0.0, 1.0)
    
    # Beta decreases as we get closer to the optimum
    # Early: beta ~ 2.0 (more exploration), Late: beta ~ 0.5 (more exploitation)
    beta = 0.5 + 1.5 * f_norm
    
    ucb = mu - beta * sigma  # Since we're minimizing, lower mu is better
    
    # Normalize both scores to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-12
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-12
    ucb_norm = (ucb_min - ucb) / ucb_range  # Invert so higher is better
    
    # Weighted combination: favor EI when close to optimum, UCB when exploring
    # Weight shifts from UCB-heavy to EI-heavy as f_best decreases
    w_ei = 0.7 + 0.3 * (1.0 - f_norm)  # 1.0 early, 0.7 late
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small bonus for points with very low sigma near low mu (exploitation)
    low_sigma_mask = sigma < np.percentile(sigma, 10)
    low_mu_mask = mu < np.percentile(mu, 20)
    exploitation_bonus = 0.1 * (low_sigma_mask & low_mu_mask).astype(float)
    
    scores = scores + exploitation_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
