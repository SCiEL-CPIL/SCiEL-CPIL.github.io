import time
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = 85.0
    
    P = 101
    D = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
        
    # We know K_min is 3.
    # We will generate subsets of size 1, 2, 3.
    # Map residue -> list of (size, terms)
    # We want the smallest size for each residue.
    
    residue_to_best_subset = {r: None for r in range(P)}
    
    # Generate subsets of size 1
    for combo in combinations(D, 1):
        res = sum(inv[x] for x in combo) % P
        # If not assigned, or if we find a smaller size (impossible here as we go 1->3)
        if residue_to_best_subset[res] is None:
            residue_to_best_subset[res] = list(combo)
            
    # Generate subsets of size 2
    for combo in combinations(D, 2):
        res = sum(inv[x] for x in combo) % P
        if residue_to_best_subset[res] is None:
            residue_to_best_subset[res] = list(combo)
            
    # Generate subsets of size 3
    for combo in combinations(D, 3):
        res = sum(inv[x] for x in combo) % P
        if residue_to_best_subset[res] is None:
            residue_to_best_subset[res] = list(combo)
            
    # Check if all residues are covered
    # If some are not, we need size 4.
    # But we assume 3 is sufficient.
    # If not, we fallback to size 4.
    
    uncovered = [r for r in range(P) if residue_to_best_subset[r] is None]
    
    if uncovered:
        # Try size 4
        for combo in combinations(D, 4):
            res = sum(inv[x] for x in combo) % P
            if residue_to_best_subset[res] is None:
                residue_to_best_subset[res] = list(combo)
                
        # Check again
        uncovered = [r for r in range(P) if residue_to_best_subset[r] is None]
        
        if uncovered:
            # Try size 5
            for combo in combinations(D, 5):
                res = sum(inv[x] for x in combo) % P
                if residue_to_best_subset[res] is None:
                    residue_to_best_subset[res] = list(combo)
                    
            uncovered = [r for r in range(P) if residue_to_best_subset[r] is None]
            
            if uncovered:
                # Fallback: use size 10 or whatever works
                # This should not happen
                for r in uncovered:
                    # Find any subset
                    for size in range(1, 11):
                        found = False
                        for combo in combinations(D, size):
                            if sum(inv[x] for x in combo) % P == r:
                                residue_to_best_subset[r] = list(combo)
                                found = True
                                break
                        if found:
                            break
    
    # Build result
    result = [None] * P
    for r in range(P):
        result[r] = residue_to_best_subset[r]
        
    return result
