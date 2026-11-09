"""Acquisition function for Bayesian optimization of weierstrass 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Diversity-penalized Expected Improvement acquisition.
    
    Combines standard EI with a penalty based on local candidate density
    to encourage exploration of distinct basins in the multi-modal landscape.
    """
    n = len(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute a diversity penalty based on local density
    # We approximate the "redundancy" of a candidate by looking at how many
    # other candidates have similar mu values (high correlation in 1D space
    # with Matern kernel roughly corresponds to proximity in input space,
    # but since we don't have input coordinates, we use mu and sigma as proxies).
    #
    # A better proxy for "distance" in the latent space of the GP posterior:
    # Candidates with very similar (mu, sigma) profiles are likely close in input space.
    # We use a simple density estimate in the (mu, sigma) space.
    
    # Normalize mu and sigma for density calculation
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    mu_range = mu_max - mu_min + 1e-9
    sigma_range = sigma_max - sigma_min + 1e-9
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Use a 2D kernel density estimate with a small bandwidth
    # For efficiency, we use a simple binning approach
    n_bins = 50
    mu_hist, _ = np.histogram(mu_norm, bins=n_bins, range=(0, 1))
    sigma_hist, _ = np.histogram(sigma_norm, bins=n_bins, range=(0, 1))
    
    # Interpolate density at each point
    mu_bin = np.clip(np.digitize(mu_norm, np.linspace(0, 1, n_bins + 1)) - 1, 0, n_bins - 1)
    sigma_bin = np.clip(np.digitize(sigma_norm, np.linspace(0, 1, n_bins + 1)) - 1, 0, n_bins - 1)
    
    # 2D density approximation using product of 1D histograms (independence assumption)
    density_2d = np.zeros(n)
    for i in range(n):
        density_2d[i] = (mu_hist[mu_bin[i]] + 1) * (sigma_hist[sigma_bin[i]] + 1)
    
    # Normalize density to [0, 1]
    density_max = np.max(density_2d)
    if density_max > 0:
        density_norm = density_2d / density_max
    else:
        density_norm = np.zeros(n)
    
    # Penalize high-density regions
    # The penalty should be proportional to the density
    # We scale the penalty so that it's significant relative to EI values
    ei_max = np.max(ei)
    if ei_max > 0:
        penalty_scale = 0.5 * ei_max
    else:
        penalty_scale = 1e-6
    
    scores = ei - penalty_scale * density_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
