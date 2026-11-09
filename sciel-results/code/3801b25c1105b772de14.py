"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Pre-compute modular inverses of 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
    
    # For each residue, find the smallest-size subset that sums to it
    # min_size[r] = smallest size, min_terms[r] = corresponding terms
    min_size = [11] * 101  # 11 means not found
    min_terms = [None] * 101
    
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        s = 0
        terms = []
        for i in range(10):
            if mask & (1 << i):
                x = i + 1
                s = (s + inv[x]) % 101
                terms.append(x)
        if size < min_size[s]:
            min_size[s] = size
            min_terms[s] = terms
    
    # All 101 residues should be covered; build result
    result = []
    for r in range(101):
        if min_terms[r] is not None:
            result.append(min_terms[r])
        else:
            # Fallback: find any subset (shouldn't happen)
            result.append([1])
    
    return result
# EVOLVE-BLOCK-END
