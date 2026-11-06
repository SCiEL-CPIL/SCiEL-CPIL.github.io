"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    
    p = 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    min_len = [11] * p
    best_subsets = [None] * p
    
    # Precompute all combinations and their sums to avoid repeated calculation
    # and to allow quick lookup of min size.
    # Since 2^10 is small, we can just iterate masks.
    # However, we want to find the MINIMUM size for each residue.
    
    # We can iterate by size to stop early if all are covered.
    for size in range(1, 11):
        if time.time() > deadline:
            break
        
        all_covered = all(ml < 11 for ml in min_len)
        if all_covered:
            break
        
        for comb in combinations(range(1, 11), size):
            s = 0
            for x in comb:
                s += inv[x]
            s %= p
            if size < min_len[s]:
                min_len[s] = size
                best_subsets[s] = list(comb)
    
    if any(bs is None for bs in best_subsets):
        for r in range(p):
            if best_subsets[r] is None:
                # Fallback
                found = False
                for size in range(1, 11):
                    for comb in combinations(range(1, 11), size):
                        s = sum(inv[x] for x in comb) % p
                        if s == r:
                            best_subsets[r] = list(comb)
                            found = True
                            break
                    if found:
                        break
    
    return [best_subsets[r] for r in range(p)]
# EVOLVE-BLOCK-END
