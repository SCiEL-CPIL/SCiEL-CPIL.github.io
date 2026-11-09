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
# Secondary heuristic: weighted zero-position bias.
# Assign higher weights to positions where the vector contains the digit 0,
# leveraging the fact that 0 is the additive identity and vectors with
# more zeros are structurally less likely to form the symmetric triples
# required for 3-term arithmetic progressions.
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
    3. Weighted zero-position bias: rewards vectors with zeros at high-weight
       positions, leveraging the additive identity property of 0 in F_3
    4. Variance (tertiary): higher variance across coordinates helps avoid APs
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
    
    # Weighted zero-position bias: assign higher weights to positions where
    # vector[i] == 0. Use a linear weight scheme where position i has weight
    # (i + 1), so later positions get higher weight. This creates a structural
    # bias that favors vectors with zeros concentrated in later positions,
    # which may reduce the likelihood of forming symmetric triples.
    zero_weight_sum = sum((i + 1) for i, v in enumerate(vector) if v == 0)
    max_zero_weight = n * (n + 1) / 2.0
    zero_weight_norm = zero_weight_sum / max_zero_weight if max_zero_weight > 0 else 0
    
    # Combine: primary is quartic balance penalty (lower is better),
    # secondary is quadratic balance (lower is better),
    # tertiary is weighted zero-position bias (higher is better),
    # quaternary is variance (higher is better),
    # quinary is sum (lower is better for tie-breaking).
    # Scale factors chosen to ensure clear tier separation.
    
    # To make higher priority = better, we negate the penalties.
    # balance_penalty4: lower is better -> -balance_penalty4 * 1000
    # balance_penalty2: lower is better -> -balance_penalty2 * 10
    # zero_weight_norm: higher is better -> +zero_weight_norm * 3
    # variance: higher is better -> +variance * 1
    # sum: lower is better -> -sum * 0.001
    
    return (-balance_penalty4 * 1000.0 
            - balance_penalty2 * 10.0 
            + zero_weight_norm * 3.0 
            + variance * 1.0 
            - s * 0.001)
