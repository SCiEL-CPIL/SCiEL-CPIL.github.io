"""Cap set greedy priority function."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Assign priority to a vector in {0,1,2}^n for greedy cap set construction.
    
    Heuristic: prefer vectors with balanced coordinate distributions,
    as they are less likely to form 3-term arithmetic progressions.
    """
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        elif v == 2:
            count_2 += 1
    
    # Prefer balanced vectors: maximize the minimum count
    min_count = min(count_0, count_1, count_2)
    
    # Penalize imbalance
    imbalance = abs(count_0 - count_1) + abs(count_1 - count_2) + abs(count_0 - count_2)
    
    return float(min_count) - 0.01 * float(imbalance)
# EVOLVE-BLOCK-END
