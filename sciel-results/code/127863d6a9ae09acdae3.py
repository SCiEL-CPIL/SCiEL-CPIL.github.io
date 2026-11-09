"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Quantile-normalized Expected Improvement (EI) acquisition function.
    
    Computes EI for all candidates, then maps the raw EI values to their
    empirical percentiles within the current candidate set. This creates a
    scale-invariant ranking that prevents domination by outlier high-variance
    points while maintaining EI's exploitation bias.
    
    Args:
        mu: array of shape (2000,) - GP posterior means
        sigma: array of shape (2000,) - GP posterior standard deviations
        f_best: float - best (lowest) observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Compute standard Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei_raw = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip tiny negative values from numerical noise
    ei_raw = np.maximum(ei_raw, 0.0)
    
    # Map to empirical percentiles (quantile-normalized)
    # This makes the score scale-invariant and robust to outliers
    # Use rank-based normalization: percentile rank of each EI value
    n = len(ei_raw)
    if n <= 1:
        return ei_raw.copy()
    
    # Get ranks (handling ties by using average rank)
    order = np.argsort(ei_raw)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    
    # Handle ties: assign average rank to tied values
    sorted_ei = ei_raw[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_ei[j + 1] == sorted_ei[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0  # 1-indexed average
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
        i = j + 1
    
    # Normalize ranks to [0, 1] percentile range
    # rank n -> 1.0, rank 1 -> 0.0
    percentile_scores = (ranks - 0.5) / n
    
    # Return percentile scores (higher = better candidate)
    # This is scale-invariant and prevents outlier domination
    return percentile_scores
# EVOLVE-BLOCK-END
