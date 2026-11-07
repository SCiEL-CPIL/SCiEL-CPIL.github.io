"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) based acquisition function.
    
    For minimization, we want P(f(x) < f_best).
    Given posterior N(mu, sigma^2), P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    
    We scale by the expected improvement magnitude to create a smooth score
    that balances exploration (high sigma) and exploitation (low mu).
    
    The score is: POI * max(0, f_best - mu) + small_bonus * sigma
    This ensures:
    - Points with high probability of improvement AND large expected gain score high
    - The sigma term provides a mild exploration bonus
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Expected improvement magnitude (positive when mu < f_best)
    expected_gain = np.maximum(f_best - mu, 0.0)
    
    # Score: POI weighted by expected gain, plus a small exploration bonus
    # The exploration bonus is scaled to be small relative to typical gains
    exploration_bonus = 0.01 * sigma_safe
    
    score = poi * expected_gain + exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
