"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a Shellsort network for N=18.
    Gaps: 9, 4, 2, 1.
    Total comparators: (18-9) + (18-4) + (18-2) + (18-1) = 9 + 14 + 16 + 17 = 56.
    """
    comparators = []
    gaps = [9, 4, 2, 1]
    
    for gap in gaps:
        for i in range(N - gap):
            comparators.append((i, i + gap))
            
    return comparators
# EVOLVE-BLOCK-END
