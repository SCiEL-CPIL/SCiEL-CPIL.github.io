"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull distribution parameters for the generated sequences
# Based on typical bin packing benchmarks with capacity 100
# Mean item size is often around 20-30.
# Let's assume shape k=1.5 and scale lambda such that mean is ~25.
# Mean = lambda * Gamma(1 + 1/k)
# Gamma(1 + 1/1.5) = Gamma(1.666) ~ 0.9027
# lambda = 25 / 0.9027 ~ 27.7
WEIBULL_SHAPE = 1.5
WEIBULL_SCALE = 27.7
CAPACITY = 100.0

# The "optimal" gap size in a thermodynamic sense is related to the 
# typical item size. If we pack items of size ~mu, the remaining gaps 
# should ideally be small, but not zero (to avoid fragmentation of future 
# medium items). 
# A common heuristic for "optimal" remaining space is related to the 
# variance or the mean itself. 
# Let's define the target gap r* as a fraction of the mean item size.
# If we pack an item, the remaining space r should be such that it's 
# likely to be filled by future items.
# The "free energy" F(r) = E[r] + T*S(r).
# Minimizing F(r) is equivalent to maximizing the likelihood of the gap 
# being "good".
# A good gap is one that is small but not too small (to allow for 
# future small items).
# Let's use the KL divergence between the empirical gap distribution 
# and a target distribution.
# Since we don't have the full distribution, we approximate the "cost" 
# of a gap r by its distance from an ideal gap r*.
# The ideal gap r* is estimated as the mean item size divided by a 
# factor, or simply the mean item size itself.
# Actually, a better approach: The "pressure" or "energy" of a gap r 
# is high if r is very large (wasted space) or very small (fragmentation 
# risk for future items that might not fit).
# However, for online bin packing, Best Fit (minimizing r) is usually 
# strong. Worst Fit (maximizing r) is strong for preserving large bins.
# The thermodynamic view suggests a balance.
# Let's define the score as -KL(r || r*), where r* is the mode of the 
# Weibull distribution or a related statistic.
# Mode of Weibull(k, lambda) = lambda * ((k-1)/k)^(1/k) for k>1.
# k=1.5, mode = 27.7 * (0.5/1.5)^(1/1.5) = 27.7 * (0.333)^(0.666) 
# = 27.7 * 0.48 = 13.3.
# So the most frequent item size is ~13.3.
# If we place an item, the remaining gap r should ideally be close to 
# this mode, so that it can be filled by a typical future item.
# Thus, we score a bin by how close (bins - item) is to the mode.
# Score = - (r - mode)^2 / (2 * sigma^2)  (Gaussian approximation of KL)
# Or simply Score = - |r - mode|

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy: Thermodynamic Gap Equilibrium
    - Models the bin packing as a system seeking minimum free energy.
    - The "energy" of a gap r is minimized when r is close to the 
      "typical" item size (mode of the Weibull distribution).
    - This encourages filling bins to a level where the remaining space 
      is just right for the next typical item, balancing fragmentation 
      and waste.
    """
    n = len(bins)
    scores = np.full(n, -1e18)
    
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_mask]
    
    # Theoretical optimal gap size (mode of the Weibull distribution)
    # k=1.5, lambda=27.7
    # mode = lambda * ((k-1)/k)^(1/k)
    k = WEIBULL_SHAPE
    lam = WEIBULL_SCALE
    if k > 1:
        mode = lam * ((k - 1) / k) ** (1.0 / k)
    else:
        mode = lam # For k<=1, mode is 0, but let's use mean as fallback
    
    # The "free energy" or cost function:
    # We want to minimize the deviation from the optimal gap.
    # A quadratic cost function is a smooth surrogate for KL divergence 
    # around the mean/mode.
    # Cost = (r - mode)^2
    # Score = -Cost
    
    # To add some robustness, we can also penalize very small gaps 
    # (fragmentation) and very large gaps (waste) more heavily if they 
    # deviate significantly.
    # The quadratic term handles this naturally.
    
    deviation = valid_remaining - mode
    cost = deviation * deviation
    
    # Score is negative cost
    scores[valid_indices] = -cost
    
    return scores
# EVOLVE-BLOCK-END
