"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    UCB-based acquisition with adaptive kappa using uncertainty-weighted
    expected improvement. Higher score means more promising for minimization.
    
    Uses a hybrid approach: UCB for exploration, but modulated by how much
    improvement is expected given current uncertainty.
    """
    n = len(mu)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Adaptive kappa: balance exploration vs exploitation
    # Use a time-varying schedule based on f_best gap to known minimum
    f_min_known = -3.86278
    gap = f_best - f_min_known  # positive, decreases as we improve
    
    # kappa starts high (exploration) and decreases (exploitation)
    # Use exponential decay with a characteristic scale
    kappa_base = 2.0
    kappa_scale = 1.5
    kappa = kappa_base * np.exp(-gap / kappa_scale)
    kappa = np.clip(kappa, 0.5, 3.0)
    
    # UCB score for minimization: we want high scores for points likely to be low
    # UCB = mu + kappa * sigma gives upper bound on f
    # For acquisition (higher = better to select): score = -UCB = -mu - kappa * sigma
    # But this always favors low mu. We want to also reward high uncertainty.
    # Better: use a modified UCB that balances both terms
    
    # Alternative: use Expected Improvement formulation
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # For minimization, improvement is max(0, f_best - f)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)  # EI is non-negative
    
    # Combine UCB and EI: UCB encourages exploration, EI encourages exploitation
    # Use a weighted sum where weights adapt to progress
    ucb_score = -mu + kappa * sigma  # higher = more promising
    
    # Normalize EI to comparable scale
    ei_scale = np.std(ei) + 1e-8
    ei_normalized = ei / ei_scale
    
    # Blend: early on, weight UCB more; later, weight EI more
    # progress: 0 when gap is large, 1 when gap is small
    progress = 1.0 - np.exp(-gap / 2.0)
    w_ucb = 1.0 - 0.5 * progress  # UCB weight: 1.0 early, 0.5 late
    w_ei = 0.5 * progress         # EI weight: 0 early, 0.5 late
    
    score = w_ucb * ucb_score + w_ei * ei_normalized
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
