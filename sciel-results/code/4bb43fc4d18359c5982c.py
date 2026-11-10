"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill
    
    MOD = 101
    DENOMS = list(range(1, 11))
    
    # Precompute inverses
    invs = [pow(x, -1, MOD) for x in DENOMS]
    
    # Generate all non-empty subsets with their sum mod 101
    # There are 2^10 - 1 = 1023 subsets
    # We want the minimum length subset for each residue
    min_len_for_residue = [11] * MOD  # 11 is larger than max possible length 10
    best_subset_for_residue = [None] * MOD
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for i in range(10):
            if mask & (1 << i):
                terms.append(DENOMS[i])
                s += invs[i]
        s %= MOD
        length = len(terms)
        
        if length < min_len_for_residue[s]:
            min_len_for_residue[s] = length
            best_subset_for_residue[s] = terms
    
    # Check if all residues are covered
    # If any residue has min_len 11, it means no subset was found (should not happen)
    # In that case, we might need to fallback, but the problem guarantees coverage.
    
    # Build the result
    result = []
    for r in range(MOD):
        if best_subset_for_residue[r] is not None:
            result.append(list(best_subset_for_residue[r]))
        else:
            # Fallback: this should not happen
            result.append([])
            
    return result
# EVOLVE-BLOCK-END
