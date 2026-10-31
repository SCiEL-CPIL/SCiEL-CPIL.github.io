import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    
    Uses a combination of structural bias (fewer 1s preferred) and
    deterministic pseudo-random tiebreaking to explore the space
    more effectively than pure lexicographic or sum-based ordering.
    """
    count_ones = sum(1 for v in vector if v == 1)
    
    # Deterministic hash for tiebreaking
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 1) + 17) & 0xFFFFFFFF
    
    # Strong bias toward fewer 1s, random within groups
    return -(count_ones * 1000 + (h % 100000))
# EVOLVE-BLOCK-END
