"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd

def construct():
    N = 256
    selected = []
    for x in range(N, 0, -1):
        # Check if x is coprime with all already selected numbers
        is_coprime_with_all = True
        for y in selected:
            if gcd(x, y) != 1:
                is_coprime_with_all = False
                break
        if is_coprime_with_all:
            selected.append(x)
    
    # The problem requires returning a list of ints.
    # The order doesn't matter for the sum, but we return the list.
    return selected
# EVOLVE-BLOCK-END
