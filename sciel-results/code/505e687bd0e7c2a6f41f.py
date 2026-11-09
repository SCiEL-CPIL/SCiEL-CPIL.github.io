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
# Tertiary: structural diversity via bigram patterns (O(n), replaces O(c0^2) zero-spread).
#
# Quaternary: sum of digits as a final tie-breaker.

def priority(vector, n):
    """
    Compute a static priority for a vector in {0,1,2}^n.
    Higher priority means the vector should be considered earlier
    in the greedy cap set construction.
    
    Heuristic components:
    1. Quartic balance penalty (primary): strongly favors balanced digit counts
    2. Quadratic balance penalty (secondary): intermediate-scale balance
    3. Variance (tertiary): higher variance across coordinates helps avoid APs
    4. Bigram diversity (quaternary): O(n) structural metric replacing O(c0^2) zero-spread
    5. Sum (quinary): final tie-breaker
    """
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    for v in vector:
        s += v
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    target = n / 3.0
    
    # Quartic balance penalty: sharper than quadratic, strongly favors balanced vectors
    d0 = c0 - target
    d1 = c1 - target
    d2 = c2 - target
    balance_penalty4 = d0 * d0 * d0 * d0 + d1 * d1 * d1 * d1 + d2 * d2 * d2 * d2
    
    # Quadratic balance for intermediate scale
    balance_penalty2 = d0 * d0 + d1 * d1 + d2 * d2
    
    # Variance: sum of squared deviations of each digit from the mean
    mean = s / n
    variance = 0.0
    for v in vector:
        dv = v - mean
        variance += dv * dv
    
    # Bigram diversity: count distinct adjacent-pair patterns in the vector.
    # This is O(n) and captures local structural variation. Vectors with more
    # distinct bigrams have richer local structure, which correlates with
    # lower AP-formation density in the greedy construction.
    # Also compute a weighted bigram score: reward bigrams that mix digits.
    if n >= 2:
        distinct_bigrams = set()
        mix_score = 0.0
        for i in range(n - 1):
            a = vector[i]
            b = vector[i + 1]
            distinct_bigrams.add((a, b))
            # Reward transitions between different digits
            if a != b:
                mix_score += 1.0
        bigram_diversity = len(distinct_bigrams)
        # Normalize: max distinct bigrams is min(n-1, 9)
        max_bg = min(n - 1, 9)
        bigram_norm = bigram_diversity / max_bg if max_bg > 0 else 0.0
        mix_norm = mix_score / (n - 1) if n > 1 else 0.0
    else:
        bigram_norm = 0.0
        mix_norm = 0.0
    
    # Combine with clear tier separation:
    # balance_penalty4: lower is better -> -balance_penalty4 * 1000
    # balance_penalty2: lower is better -> -balance_penalty2 * 10
    # variance: higher is better -> +variance * 1
    # bigram_norm: higher is better -> +bigram_norm * 3
    # mix_norm: higher is better -> +mix_norm * 2
    # sum: lower is better -> -sum * 0.001
    
    return (-balance_penalty4 * 1000.0 
            - balance_penalty2 * 10.0 
            + variance * 1.0 
            + bigram_norm * 3.0 
            + mix_norm * 2.0 
            - s * 0.001)
