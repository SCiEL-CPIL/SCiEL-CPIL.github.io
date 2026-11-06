import time

def construct():
    # The problem requires 101 non-empty lists of distinct denominators from 1..10
    # such that the sum of their modular inverses modulo 101 equals the row index.
    # We want to minimize the largest row length.
    # Since denominators can recur across rows, each row is independent.
    # Thus, the optimal strategy is to find the smallest possible subset for each residue.
    
    # Precompute inverses of 1..10 modulo 101
    invs = [pow(x, -1, 101) for x in range(1, 11)]
    denominators = list(range(1, 11))
    
    # Precompute all non-empty subsets: (sum_mod_101, size, denom_list)
    all_subsets = []
    for mask in range(1, 1 << 10):
        denom_list = []
        s = 0
        for i in range(10):
            if mask & (1 << i):
                denom_list.append(denominators[i])
                s = (s + invs[i]) % 101
        all_subsets.append((s, len(denom_list), denom_list))
    
    # Sort by size, then by sum (for deterministic tie-breaking)
    all_subsets.sort(key=lambda x: (x[1], x[0]))
    
    # For each residue, find the smallest-size subset that covers it
    best_rep = {}  # residue -> denom_list
    
    for s, sz, dl in all_subsets:
        if s not in best_rep:
            best_rep[s] = dl
    
    # Construct the result
    # All residues 0..100 should be covered by subsets of size <= 3 (likely)
    # If any are missing (which shouldn't happen), we fall back to any valid subset
    result = []
    for r in range(101):
        if r in best_rep:
            result.append(best_rep[r])
        else:
            # Fallback: find any subset (should not happen for p=101, d=10)
            # Use the largest subset as a safe bet for existence
            result.append(denominators)
            
    return result
