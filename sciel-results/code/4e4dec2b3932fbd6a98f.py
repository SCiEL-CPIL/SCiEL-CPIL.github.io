"""Cap set greedy construction with a refined static priority heuristic."""
import time
import math
import random
from itertools import product

# We use a refined static priority heuristic for cap set construction.
# A 3-term AP in F_3^n is x, y, z with x + z = 2y (mod 3).
# The priority function must be static (no knowledge of the current set).
#
# Primary heuristic: quartic balance penalty (sharper than quadratic).
# For a vector v, let c0, c1, c2 be the counts of digits 0, 1, 2.
# penalty4 = (c0 - n/3)^4 + (c1 - n/3)^4 + (c2 - n/3)^4
# The quartic power more strongly penalizes imbalanced vectors,
# creating a sharper preference for near-balanced digit distributions.
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
    
    Heuristic components:
    1. Quartic balance penalty (primary): strongly favors balanced digit counts
    2. Quadratic balance penalty (secondary): intermediate-scale balance
    3. Variance (tertiary): higher variance across coordinates helps avoid APs
    4. Zero-position spread (new): rewards vectors where zeros are spread
       across different coordinates, as 0 is the additive identity in F_3
       and isolated zeros are less likely to participate in symmetric triples
    5. Sum (quaternary): final tie-breaker
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # Quartic balance penalty: sharper than quadratic, strongly favors balanced vectors
    balance_penalty4 = (c0 - target) ** 4 + (c1 - target) ** 4 + (c2 - target) ** 4
    
    # Also keep quadratic for intermediate scale
    balance_penalty2 = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Pairwise variance: sum of squared deviations of each digit from the mean
    s = sum(vector)
    mean = s / n
    variance = sum((v - mean) ** 2 for v in vector)
    
    # Zero-position spread: for each coordinate i where vector[i] == 0,
    # count how many other coordinates j != i also have vector[j] == 0.
    # A vector with zeros spread out (fewer collisions) gets higher score.
    # We compute: for each zero at position i, the number of non-zero positions
    # is (n - c0). The "isolation" of a zero at position i is the count of
    # positions j where vector[j] != 0. But since that's constant for all zeros,
    # instead we measure the pairwise distance between zero positions.
    # 
    # Better approach: sum over all pairs of zero positions (i,j) of |i-j|.
    # This rewards zeros that are far apart from each other.
    zero_positions = [i for i, v in enumerate(vector) if v == 0]
    if len(zero_positions) >= 2:
        # Sum of pairwise distances between zero positions
        zero_spread = sum(abs(zero_positions[i] - zero_positions[j]) 
                         for i in range(len(zero_positions)) 
                         for j in range(i+1, len(zero_positions)))
        # Normalize by max possible spread
        max_possible = len(zero_positions) * (len(zero_positions) - 1) * (n - 1) / 2
        zero_spread_norm = zero_spread / max_possible if max_possible > 0 else 0
    else:
        zero_spread_norm = 0
    
    # Combine: primary is quartic balance penalty (lower is better),
    # secondary is quadratic balance (lower is better),
    # tertiary is variance (higher is better),
    # quaternary is zero spread (higher is better),
    # quinary is sum (lower is better for tie-breaking).
    # Scale factors chosen to ensure clear tier separation.
    
    # To make higher priority = better, we negate the penalties.
    # balance_penalty4: lower is better -> -balance_penalty4 * 1000
    # balance_penalty2: lower is better -> -balance_penalty2 * 10
    # variance: higher is better -> +variance * 1
    # zero_spread_norm: higher is better -> +zero_spread_norm * 5
    # sum: lower is better -> -sum * 0.001
    
    return (-balance_penalty4 * 1000.0 
            - balance_penalty2 * 10.0 
            + variance * 1.0 
            + zero_spread_norm * 5.0 
            - s * 0.001)
