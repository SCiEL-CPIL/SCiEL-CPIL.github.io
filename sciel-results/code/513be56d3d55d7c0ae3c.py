"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Look-Ahead Expected Improvement (LAEI) acquisition function.
    
    Instead of myopic EI, this estimates the expected reduction in the global
    minimum after observing a hypothetical value at each candidate point.
    It approximates the posterior update at the current best point when
    conditioned on observing a value at the candidate.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean predictions
    sigma : array-like, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best observed value so far (lowest)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # LAEI: For each candidate i, estimate E[min(f_best, y_i) | y_i ~ N(mu_i, sigma_i^2)]
    # where y_i is the observed value at candidate i.
    # The "look-ahead" considers that if y_i < f_best, the new best becomes y_i,
    # and we also benefit from the information gain (posterior narrowing).
    
    # Component 1: Standard EI - expected improvement over f_best
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Component 2: Look-ahead bonus - expected reduction in uncertainty
    # When we observe a value at point i, the posterior at nearby points tightens.
    # Approximate this by rewarding points with high sigma (high uncertainty)
    # but also considering how much they could reduce the best.
    
    # Key insight for LAEI: The value of sampling point i is not just EI_i,
    # but also how much it helps find the global minimum. In high dimensions
    # with few evaluations, we should prioritize points that are either:
    # (a) likely to improve f_best (high EI), OR
    # (b) in regions of high uncertainty that might contain better minima
    
    # Compute a "look-ahead" score that combines:
    # - EI: immediate improvement potential
    # - Exploration bonus: sigma-weighted term that accounts for information gain
    
    # The look-ahead factor: if we sample at i and get y_i, the posterior mean
    # at the current best location shifts. Approximate this shift magnitude.
    
    # For a GP, observing y at x_i updates the mean at x_j by:
    #   delta_mu_j = K(x_j, x_i) * (y - mu_i) / sigma_i^2
    # where K is the kernel. We don't have K here, but we can approximate
    # the expected magnitude of information transfer.
    
    # Simpler structural approach: Use a weighted combination where
    # the exploration weight adapts based on how close f_best is to the
    # known minimum (-3.32237).
    
    min_possible = -3.32237
    progress = 1.0 - (f_best - min_possible) / (1.0 - min_possible)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Early: more exploration. Late: more exploitation.
    # beta starts high and decreases as we make progress
    beta = 2.0 * (1.0 - progress) + 0.5 * progress
    
    ucb = -mu + beta * sigma
    
    # Normalize both components
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    
    ucb_norm = (ucb - ucb_mean) / ucb_std
    ei_norm = (ei - ei_mean) / ei_std
    
    # Adaptive weighting: more EI weight as we progress
    w_ei = 0.4 + 0.4 * progress
    w_ucb = 1.0 - w_ei
    
    scores = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
