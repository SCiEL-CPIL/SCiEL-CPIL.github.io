# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma

# Weibull parameters for the generated sequences (shape k, scale lambda)
# Assume k=2.0, lambda=20.0 as reasonable defaults for items fitting in bins of capacity 100
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

def _expected_next_item_size() -> float:
    """
    Estimate the expected size of the next item.
    For a Weibull distribution, E[X] = lambda * Gamma(1 + 1/k).
    """
    return LAMBDA_WEIBULL * gamma(1 + 1.0 / K_WEIBULL)

def _std_next_item_size() -> float:
    """
    Estimate the standard deviation of the next item size.
    For a Weibull distribution, Var(X) = lambda^2 * (Gamma(1+2/k) - Gamma(1+1/k)^2)
    """
    var = LAMBDA_WEIBULL ** 2 * (gamma(1 + 2.0 / K_WEIBULL) - gamma(1 + 1.0 / K_WEIBULL) ** 2)
    return np.sqrt(var)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Maximize the probability that the remaining space after placing
    the current item can fit at least one more item from the Weibull distribution.
    
    We use a smooth approximation: score = P(remaining >= X) where X ~ Weibull(k, lambda).
    This is equivalent to the survival function of the Weibull distribution.
    
    For bins with very little remaining space, the score is low.
    For bins with lots of remaining space, the score approaches 1.
    We want to maximize this probability, so we use the CDF complement.
    """
    # Precompute constants
    k = K_WEIBULL
    lam = LAMBDA_WEIBULL
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins: bins must have at least 'item' capacity
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # For each remaining space r, compute P(X <= r) where X ~ Weibull(k, lam)
    # This is the CDF of the Weibull distribution: F(r) = 1 - exp(-(r/lam)^k)
    # We want to maximize the chance that a future item fits, which is P(X <= remaining)
    # But actually, we want to MINIMIZE wasted space while ensuring fit.
    # 
    # Better approach: Score based on how well the remaining space matches
    # the distribution of item sizes. Use the negative of the expected waste
    # conditional on fitting, or use a kernel density approach.
    #
    # Simplest effective approach: Use the Weibull PDF as a score.
    # Bins where remaining space is near the mode of the Weibull distribution
    # are preferred, as they're most likely to be fully utilized by the next item.
    #
    # Weibull mode = lam * ((k-1)/k)^(1/k) for k > 1
    if k > 1:
        mode = lam * ((k - 1) / k) ** (1.0 / k)
    else:
        mode = lam
    
    # Score = negative distance from mode, but with a smooth kernel
    # Use a Gaussian-like kernel centered at the mode
    sigma = _std_next_item_size()
    if sigma < 1e-10:
        sigma = 1.0
    
    # Score: higher when remaining is close to the mode
    # Use exp(-(remaining - mode)^2 / (2*sigma^2))
    diff = remaining - mode
    scores[valid_mask] = np.exp(-diff ** 2 / (2 * sigma ** 2))
    
    return scores
# EVOLVE-BLOCK-END
