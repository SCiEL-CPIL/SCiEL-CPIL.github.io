"""Cap set greedy construction with a refined static priority heuristic."""
import time
import math
import random
from itertools import product

# We use a refined static priority heuristic for cap set construction.
# A 3-term AP in F_3^n is x, y, z with x + z = 2y (mod 3).
# The priority function must be static (no knowledge of the current set).
#
# Primary heuristic: balance penalty.
# For a vector v, let c0, c1, c2 be the counts of digits 0, 1, 2.
# penalty = (c0 - n/3)^2 + (c1 - n/3)^2 + (c2 - n/3)^2
# Lower penalty means more balanced, which tends to avoid APs.
#
# Secondary heuristic: pairwise digit variance.
# For each coordinate, the "spread" of digits matters. Vectors with
# more variation across coordinates are less likely to complete APs
# with other similar vectors. We compute the sum of squared deviations
# of each coordinate from the mean of the vector's digits.
#
# Tertiary: sum of digits as a final tie-breaker.

def priority(vector, n):
    """
    Compute a static priority for a vector in {0,1,2}^n.
    Higher priority means the vector should be considered earlier
    in the greedy cap set construction.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    balance_penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Pairwise variance: sum of squared deviations of each digit from the mean
    s = sum(vector)
    mean = s / n
    variance = sum((v - mean) ** 2 for v in vector)
    
    # Combine: primary is balance penalty (lower is better),
    # secondary is variance (higher is better, as more spread helps avoid APs),
    # tertiary is sum (lower is better for tie-breaking).
    # Scale factors chosen to ensure primary dominates, then secondary, then tertiary.
    # balance_penalty ranges from 0 to ~n (max when all digits are the same: n*(n/3)^2*2 + n*(2n/3)^2 = ... let's compute max)
    # For n=3: max penalty = 6 (all same digit). Min = 0 (balanced).
    # variance ranges from 0 to ~n (max when half are 0 and half are 2, etc.)
    # s ranges from 0 to 2n.
    
    # Use exponential scaling to ensure clear separation between tiers.
    # Tier 1: balance penalty (weight ~1000)
    # Tier 2: variance (weight ~10, inverted since higher variance is better)
    # Tier 3: sum (weight ~1, inverted since lower sum is better)
    
    # To make higher priority = better, we negate the penalties.
    # balance_penalty: lower is better -> -balance_penalty * 1000
    # variance: higher is better -> +variance * 10
    # sum: lower is better -> -sum * 1
    
    return -balance_penalty * 1000.0 + variance * 10.0 - s
