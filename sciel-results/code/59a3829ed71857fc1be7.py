# EVOLVE-BLOCK-START
import numpy as np

# Weibull parameters for the generated sequences (shape k, scale lambda)
# Assuming k=2.0, lambda=20.0 based on previous analysis and typical benchmarks.
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Maximize the probability that the *next* item (drawn from the Weibull
    distribution) fits into the remaining space of the bin after placing the current item.
    
    Let R be the remaining space after placing the current item.
    The probability that a future item Y fits is P(Y <= R) = CDF_Y(R).
    For Weibull(k, lambda), CDF(y) = 1 - exp(-(y/lambda)^k).
    
    We assign the score as this probability. Higher probability means higher score.
    This effectively implements a Worst Fit strategy, as the CDF is monotonically
    increasing with R, so bins with the most remaining space will have the highest
    probability of accepting the next item.
    """
    # Calculate remaining space if the item is placed in each bin
    remaining = bins - item
    
    # Initialize scores to -infinity for invalid bins (where item doesn't fit)
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Identify valid bins: remaining space must be non-negative
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate probability for valid bins
    R_valid = remaining[valid_mask]
    
    # P(Y <= R) = 1 - exp(-(R/lambda)^k)
    # Using vectorized operations
    # Note: For very large R, (R/lambda)^k can be large, exp(-large) -> 0, prob -> 1.
    # This is stable.
    scaled_R = R_valid / LAMBDA_WEIBULL
    exponent = -(scaled_R ** K_WEIBULL)
    prob = 1.0 - np.exp(exponent)
    
    scores[valid_mask] = prob
    
    return scores
# EVOLVE-BLOCK-END
