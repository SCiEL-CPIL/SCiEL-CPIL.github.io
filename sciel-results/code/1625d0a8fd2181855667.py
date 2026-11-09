"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition: maximize the predicted 5th percentile
    of the posterior distribution.

    For minimization, a lower 5th percentile indicates a higher chance of
    finding a very low function value. We score by -mu - q * sigma where
    q = z_{0.05} ≈ -1.645, so score = -mu + 1.645 * sigma.

    This is structurally different from the adaptive LCB: it uses a fixed,
    aggressive lower-tail quantile rather than a progress-scheduled beta.
    """
    # Ensure sigma is non-negative and handle near-zero values
    sigma = np.maximum(sigma, 1e-10)

    # 5th percentile quantile: z_{0.05} ≈ -1.645
    # Predicted 5th percentile = mu + z_{0.05} * sigma = mu - 1.645 * sigma
    # We want to maximize the chance of a low value, so we minimize the
    # predicted 5th percentile. Score = -predicted_5th_percentile
    # score = -(mu - 1.645 * sigma) = -mu + 1.645 * sigma
    score = -mu + 1.645 * sigma

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
