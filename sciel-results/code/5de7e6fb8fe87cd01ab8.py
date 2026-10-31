import math
import time

def priority(vector, n):
    """
    Direct approach for cap set construction.
    
    A cap set in {0,1,2}^n is a set of vectors with no three distinct vectors a, b, c
    such that a + b + c = 0 mod 3 (i.e., they form an arithmetic progression).
    
    Key insight: Vectors with balanced symbol counts (roughly n/3 of each 0,1,2)
    tend to be more compatible with each other in cap sets. We prioritize vectors
    that:
    1. Have symbol counts close to n/3 each (balance)
    2. Avoid patterns that create 3-term APs with common cap set elements
    
    For a direct approach, we use:
    - Primary: Balance of symbol counts (minimize deviation from n/3)
    - Secondary: A structural hash that favors vectors with diverse coordinate patterns
    """
    # Deadline guard - we must be fast since this is called many times
    # Count symbols
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Balance score: prefer vectors with counts close to n/3
    # Use sum of squared deviations (smooth surrogate) for better differentiation
    ideal = n / 3.0
    sq_dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # The more balanced, the higher the priority
    # Scale balance to dominate
    balance_score = -sq_dev * 100.0
    
    # Secondary: sum of squares of coordinate values (c1 + 4*c2)
    # This breaks symmetry and favors vectors with more "energy" in non-zero coords
    # Light weight so balance still dominates
    sq_coord = c1 + 4 * c2
    coord_score = sq_coord / n * 0.5
    
    # Additional structural priority: prefer vectors where no coordinate repeats too much
    # Use a simple hash for tie-breaking, but weight it lightly
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 7) % (10**9 + 7)
    
    hash_score = h / (10**9 + 7)
    
    return float(balance_score + coord_score + hash_score)
