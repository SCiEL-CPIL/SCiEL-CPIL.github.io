"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Compute Quantile-based acquisition (Q0.1) for minimization.
    
    This targets the 10th percentile of the posterior predictive distribution.
    For a Gaussian posterior, the 10th percentile is mu - z_0.1 * sigma,
    where z_0.1 is the 90th percentile of the standard normal distribution
    (since we want the lower tail, we subtract the positive z-score).
    
    z_0.1 (90th percentile of standard normal) is approximately 1.28155.
    
    By selecting points with the lowest quantile, we prioritize regions where
    there is a high probability of finding a value significantly below the
    current best, effectively balancing exploitation (low mu) and exploration
    (high sigma) in a way that directly optimizes for new record lows.
    """
    # z-score for the 10th percentile (lower tail)
    # P(Z < -1.28155) = 0.10
    # So, 10th percentile = mu - 1.28155 * sigma
    z_01 = 1.28155
    
    # Compute the 10th percentile of the posterior
    q01 = mu - z_01 * sigma
    
    # Ensure finite scores
    q01 = np.where(np.isfinite(q01), q01, -np.inf)
    
    return q01
# EVOLVE-BLOCK-END
