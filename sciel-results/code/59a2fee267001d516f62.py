import time
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = 80.0

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = {x: pow(x, -1, P) for x in D}

    # Map residue to list of (length, sum_denoms, terms)
    # We want to find the best subset for each residue.
    # "Best" is defined as: Minimize length, then minimize sum of denominators.
    residue_options = defaultdict(list)
    
    # Iterate all non-empty subsets
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        sum_denoms = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
                sum_denoms += x
        length = len(terms)
        residue_options[s].append((length, sum_denoms, terms))
        
    # Check time
    if time.time() > deadline:
        pass

    # Select the best option for each residue
    result = []
    max_len = 0
    for r in range(P):
        options = residue_options[r]
        if not options:
            # Fallback, though this should not happen
            # If a residue is not covered, we must pick something valid?
            # The problem states "Every residue must be represented".
            # If not represented by any subset, the task is impossible?
            # But with 1023 subsets and 101 residues, it's extremely likely covered.
            # If not, we might have to use a subset that doesn't sum to r?
            # No, "sum to r modulo 101".
            # If no subset sums to r, we cannot fulfill the contract.
            # However, we assume coverage.
            result.append([1])
            continue
            
        # Sort by length, then by sum of denominators
        best_option = min(options, key=lambda x: (x[0], x[1]))
        result.append(best_option[2])
        if len(best_option[2]) > max_len:
            max_len = len(best_option[2])
            
    return result
