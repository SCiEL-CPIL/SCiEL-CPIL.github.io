"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    deadline = time.time() + 50  # safety margin
    
    # Greedy approach: select largest integers first, maintaining pairwise coprimality
    selected = []
    
    # Try to include 1 first (it's coprime with everything)
    selected.append(1)
    
    # Iterate from largest to smallest
    for a in range(n, 1, -1):
        if time.time() > deadline:
            break
        # Check if a is coprime with all already selected
        if all(math.gcd(a, b) == 1 for b in selected):
            selected.append(a)
    
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
