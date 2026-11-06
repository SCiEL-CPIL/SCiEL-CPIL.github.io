"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The Witt design S(5,6,12) can be constructed as the set of 6-subsets
    # of 12 distinct non-zero vectors in GF(2)^4 that sum to the zero vector.
    # We use the integers 1..12 as representatives for 12 distinct non-zero vectors.
    
    from itertools import combinations
    
    points = list(range(1, 13))
    blocks = []
    
    for combo in combinations(points, 6):
        # XOR is addition in GF(2)^4
        s = 0
        for x in combo:
            s ^= x
        if s == 0:
            blocks.append(list(combo))
            
    return blocks
# EVOLVE-BLOCK-END
