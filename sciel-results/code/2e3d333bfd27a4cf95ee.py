"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    # Safety margin: The problem is tiny, so this should finish instantly.
    # 90s limit is generous.
    
    p = 101
    denominators = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in denominators:
        inv[x] = pow(x, -1, p)
        
    # We want to find the minimum K such that every residue r in 0..100
    # can be represented as a sum of inverses of a subset of size <= K.
    # Then, for each residue, we pick the smallest subset (size <= K) that represents it.
    
    # Step 1: Enumerate all non-empty subsets of {1..10}
    # Store for each residue, the smallest size subset found so far, and the subset itself.
    # min_size[r] = minimum size of a subset summing to r
    # best_subset[r] = the subset achieving min_size[r]
    
    min_size = [11] * p  # 11 is infinity (max size is 10)
    best_subset = [None] * p
    
    # We iterate by size to naturally find the smallest subsets first.
    # However, we need to know the global minimum max-size K.
    # Let's first collect all subsets grouped by size.
    
    # subsets_by_size[k] = list of (residue, subset_list)
    subsets_by_size = {k: [] for k in range(1, 11)}
    
    for size in range(1, 11):
        for comb in combinations(denominators, size):
            s = sum(inv[x] for x in comb) % p
            subsets_by_size[size].append((s, list(comb)))
            
    # Step 2: Determine the minimum K such that all residues are covered by subsets of size <= K
    covered = [False] * p
    min_k_required = 10
    
    for k in range(1, 11):
        # Add subsets of size k to the covered set
        for res, subset in subsets_by_size[k]:
            covered[res] = True
        
        # Check if all are covered
        if all(covered):
            min_k_required = k
            break
            
    # Step 3: For each residue, pick the smallest subset (size <= min_k_required)
    # Since we want to minimize the *maximum* row length, and we've established that
    # min_k_required is the theoretical lower bound for the max length,
    # we just need to ensure every row has a length <= min_k_required.
    # To be safe and potentially minimize the *average* or just be robust,
    # we pick the absolute smallest subset for each residue among those with size <= min_k_required.
    
    result = [None] * p
    
    # We can just scan all subsets of size <= min_k_required and keep the smallest for each residue.
    # Actually, since we want to minimize the max length, and the max length is fixed at min_k_required
    # (unless we can do better, which we can't by definition of min_k_required),
    # any valid assignment with max length min_k_required is optimal.
    # However, picking the smallest available for each residue is a good strategy 
    # and guarantees the max length is <= min_k_required.
    
    # Initialize best for each residue to None
    # We will iterate through sizes 1 to min_k_required
    # and update the best subset for each residue if we find a smaller one.
    # But since we iterate by increasing size, the first one found for a residue is the smallest.
    
    found_for_residue = [False] * p
    
    for k in range(1, min_k_required + 1):
        for res, subset in subsets_by_size[k]:
            if not found_for_residue[res]:
                result[res] = subset
                found_for_residue[res] = True
                
    # Safety check: if any residue wasn't found (should not happen), fallback to any subset
    for r in range(p):
        if not found_for_residue[r]:
            # Find any subset that sums to r
            for k in range(1, 11):
                for res, subset in subsets_by_size[k]:
                    if res == r:
                        result[r] = subset
                        found_for_residue[r] = True
                        break
                if found_for_residue[r]:
                    break
            if not found_for_residue[r]:
                raise RuntimeError(f"Residue {r} not covered")
                
    return result

# EVOLVE-BLOCK-END
