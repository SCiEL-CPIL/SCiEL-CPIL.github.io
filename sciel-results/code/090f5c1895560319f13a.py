"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    MOD = 101
    DENOMS = list(range(1, 11))
    
    # Precompute modular inverses
    invs = [0] * 11
    for x in DENOMS:
        invs[x] = pow(x, -1, MOD)
    
    # For each residue, find the minimum-size subset that produces it
    min_size = [11] * MOD  # 11 = infinity sentinel
    best_subset = [None] * MOD
    
    # Iterate all 1023 non-empty subsets of {1..10}
    for mask in range(1, 1 << 10):
        s = 0
        size = 0
        terms = []
        for i in range(10):
            if mask & (1 << i):
                d = i + 1
                terms.append(d)
                s = (s + invs[d]) % MOD
                size += 1
        
        if size < min_size[s]:
            min_size[s] = size
            best_subset[s] = terms
    
    # Build result: for each residue r, use its best (smallest) subset
    result = []
    for r in range(MOD):
        result.append(best_subset[r])
    
    return result
# EVOLVE-BLOCK-END
