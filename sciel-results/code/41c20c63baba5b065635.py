"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Quantile-based Expected Improvement (QEI) acquisition function.
    
    Instead of maximizing the expected improvement (which can be overly
    exploitative when uncertainty is small), we maximize the probability
    of improving by a fixed margin epsilon. This creates a more robust
    balance between exploration and exploitation.
    
    Q(x) = P(f(x) < f_best - epsilon) = Phi((f_best - epsilon - mu) / sigma)
    
    We also add a small exploration bonus based on sigma to ensure we
    don't completely ignore uncertain regions.
    """
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive epsilon: scale with the current best value
    # When f_best is close to 0 (minimum), use a smaller epsilon
    eps = 0.05 * max(abs(f_best), 1e-6)
    
    # Compute the quantile-based score
    z = (f_best - eps - mu) / sigma_safe
    qei = norm.cdf(z)
    
    # Add a small exploration bonus to avoid completely ignoring
    # high-uncertainty regions. The bonus is scaled to be small
    # relative to the QEI score.
    # Use a normalized sigma to keep the bonus in a reasonable range
    sigma_norm = sigma_safe / (sigma_safe + 0.1)
    exploration_bonus = 0.05 * sigma_norm
    
    # Combine: primarily QEI, with a small exploration component
    scores = qei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
