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
    if n > 0:
        mean = s / n
        variance = sum((v - mean) ** 2 for v in vector)
    else:
        variance = 0.0
        
    # Positional variance: sum of squared deviations of indices for each digit
    # This rewards vectors where each digit is spread out across coordinates
    pos_var = 0.0
    indices = {0: [], 1: [], 2: []}
    for i, v in enumerate(vector):
        indices[v].append(i)
        
    for d in [0, 1, 2]:
        idx_list = indices[d]
        k = len(idx_list)
        if k > 1:
            m = sum(idx_list) / k
            pos_var += sum((i - m) ** 2 for i in idx_list)
        # else 0
        
    # Combine heuristics
    # balance_penalty: O(n^2), lower is better
    # pos_var: O(n^2), higher is better
    # variance: O(n), higher is better
    # s: O(n), lower is better
    
    # Weights:
    # pos_var is generally larger than balance_penalty in magnitude for spread out vectors.
    # Let's try equal weight for balance and pos_var, scaled to be competitive.
    # Max balance_penalty ~ 2/3 n^2.
    # Max pos_var ~ n^2/12 * 3 = n^2/4 (if uniform distribution).
    # So they are comparable.
    
    return -balance_penalty * 1.0 + pos_var * 0.5 + variance * 0.1 - s
