# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma, gammainc

# Weibull parameters for the generated sequences (shape k, scale lambda)
# Assuming K=2.0, Lambda=20.0 based on typical bin packing benchmarks with capacity 100.
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

# Pre-calculate constants for the Expected Waste calculation
# E[Waste] = R - Integral_0^R x f(x) dx
# Integral_0^R x f(x) dx = Lambda * Gamma(3/K) * P(3/K, (R/Lambda)^K)
_GAMMA_3K = gamma(3.0 / K_WEIBULL)
_A_PARAM = 3.0 / K_WEIBULL

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Minimize the Expected Waste of the remaining space.
    Expected Waste = E[max(R - X, 0)] is not quite right, it's the waste left in the bin.
    If next item X fits (X <= R), waste is R - X.
    If next item X doesn't fit (X > R), waste is R (bin is effectively closed for this item, 
    though in online packing we might try other bins, but for this specific bin's future, 
    the space R is lost if we don't use it now? No, we are deciding WHERE to put the CURRENT item.
    Once placed, the remaining space R is fixed for that bin.
    The "waste" is the space in R that cannot be filled by the NEXT item.
    If X > R, the entire R is wasted (for the next item).
    If X <= R, the waste is R - X.
    
    So E[Waste] = R * P(X > R) + E[(R - X) | X <= R] * P(X <= R)
    = R * (1 - F(R)) + (R * F(R) - E[X * 1_{X<=R}])
    = R - R * F(R) + R * F(R) - E[X * 1_{X<=R}]
    = R - E[X * 1_{X<=R}]
    = R - Integral_0^R x f(x) dx
    
    We minimize this waste by maximizing -E[Waste].
    """
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # Calculate the integral term: Integral_0^R x f(x) dx
    # x_term = (R / Lambda) ^ K
    x_term = (remaining / LAMBDA_WEIBULL) ** K_WEIBULL
    # P(a, x) is the regularized lower incomplete gamma function
    p_term = gammainc(_A_PARAM, x_term)
    integral_val = LAMBDA_WEIBULL * _GAMMA_3K * p_term
    
    # Expected Waste
    expected_waste = remaining - integral_val
    
    # Score is negative expected waste
    scores[valid_mask] = -expected_waste
    
    return scores
# EVOLVE-BLOCK-END
