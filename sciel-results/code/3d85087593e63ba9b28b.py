import time
import itertools

def construct():
    start_time = time.time()
    
    P = 101
    N = 10

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, P - 2, P)

    # Initialize tracking for minimum size subset for each residue
    min_size = [float('inf')] * P
    best_subset = [None] * P

    # Iterate through subset sizes from 1 to N
    # We stop early if all residues are covered, but for N=10, 
    # iterating up to 5 is very fast and covers almost all.
    # We'll iterate up to N to be safe, but break if all covered.
    
    for k in range(1, N + 1):
        for comb in itertools.combinations(range(1, N + 1), k):
            # Calculate sum of inverses modulo P
            s = 0
            for x in comb:
                s += inv[x]
            r = s % P
            
            # Update if this is a smaller size for this residue
            if k < min_size[r]:
                min_size[r] = k
                best_subset[r] = list(comb)
        
        # Check if all residues are covered
        # This check is O(P) = 101, which is negligible
        if all(s != float('inf') for s in min_size):
            break

    # Construct the result
    result = []
    for r in range(P):
        if best_subset[r] is None:
            # Fallback: should not happen
            result.append([1])
        else:
            result.append(best_subset[r])
            
    return result
