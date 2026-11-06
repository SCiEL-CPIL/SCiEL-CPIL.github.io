"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    base = 51975
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # Sort in DECREASING order (different from seed's increasing order)
    divisors.sort(reverse=True)
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    for m in divisors:
        # For each possible residue r in [0, m), compute how many uncovered residues it covers
        best_r = 0
        best_count = -1
        for r in range(m):
            # Count uncovered residues in this class
            count = np.count_nonzero(covered[r::m] == 0)
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    
    return classes
# EVOLVE-BLOCK-END
