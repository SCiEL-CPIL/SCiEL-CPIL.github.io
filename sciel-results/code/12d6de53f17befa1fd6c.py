"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Quantile Acquisition Function (QAF) with numerical stability improvements.
    # Maximizes P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    
    # Ensure sigma is strictly positive to avoid division by zero or negative values.
    # Using a small epsilon to prevent numerical issues.
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the Z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip Z-scores to a reasonable range to avoid saturation of the CDF.
    # norm.cdf(10) is ~1.0, norm.cdf(-10) is ~0.0.
    # Clipping preserves the ordering for extreme values while ensuring
    # distinct scores for points that are "extremely likely" or "extremely unlikely".
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Compute the acquisition score
    scores = norm.cdf(z_clipped)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
