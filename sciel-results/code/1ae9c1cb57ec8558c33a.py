# EVOLVE-BLOCK-START
import time
import math

# Cache for priority computation to avoid redundant work
_priority_cache = {}

def priority(vector, n):
    """
    Prioritize vectors based on geometric conflict potential with spectral smoothness.
    
    A vector v is "safe" if it forms few affine lines with other vectors.
    In F_3^n, three points a, b, c form an affine line if a + b + c = 0 (mod 3).
    
    We use a combined heuristic:
    1. Centrality: vectors with balanced symbol counts are more central and risky
    2. Midpoint risk: vectors with many 1s are more likely to be midpoints
    3. Spectral smoothness: penalize high-frequency components (alternating patterns)
       which may create structured conflicts with other vectors
    """
    # Use a hashable key for caching
    key = (tuple(vector), n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    # Count symbols
    n1 = 0
    n2 = 0
    for x in vector:
        if x == 1:
            n1 += 1
        elif x == 2:
            n2 += 1
    n0 = n - n1 - n2
    
    # Penalty for being "central" - vectors with balanced counts are more central
    # and thus have more potential line partners
    target = n / 3.0
    centrality = (n0 - target)**2 + (n1 - target)**2 + (n2 - target)**2
    # Lower centrality = more balanced = MORE risky, so we want to PENALIZE low centrality
    
    # Additional penalty: vectors with many 1s are more likely to be midpoints
    # because 1 is the "middle" value in F_3
    midpoint_risk = n1 * 2.0
    
    # Spectral smoothness: compute high-frequency energy
    # A vector with alternating 0,1,2,0,1,2... has high frequency
    # We measure this by looking at consecutive differences
    if n > 1:
        # Count sign changes in the sequence (treating 0->1->2->0 as cyclic)
        # High frequency = many transitions between different values
        transitions = 0
        for i in range(n - 1):
            if vector[i] != vector[i+1]:
                transitions += 1
        # Normalize: max transitions is n-1
        freq_ratio = transitions / (n - 1) if n > 1 else 0
        # Penalize high frequency (alternating patterns are riskier)
        spectral_penalty = freq_ratio * 1.5
    else:
        spectral_penalty = 0
    
    # Variance-based risk
    variance = (n0 * 0 + n1 * 1 + n2 * 2) / n if n > 0 else 0
    variance = (n0 * 0 + n1 * 1 + n2 * 4) / n - variance**2 if n > 0 else 0
    
    # Combined score: we want to MINIMIZE line formation potential
    # Risk increases with: low centrality (balanced), high midpoint risk, high frequency
    # So we return negative risk
    risk = -centrality * 0.5 + midpoint_risk * 1.0 + spectral_penalty - variance * 0.3
    result = -risk
    
    _priority_cache[key] = result
    return result
# EVOLVE-BLOCK-END
