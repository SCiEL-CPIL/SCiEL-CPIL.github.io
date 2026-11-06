"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Precompute modular inverses
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # Generate all non-empty subsets
    # Subset represented by mask 1..1023
    # For each mask, compute sum and size
    subsets_by_residue = {r: [] for r in range(101)}
    
    for mask in range(1, 1 << 10):
        total = 0
        size = 0
        terms = []
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                total += inv[x]
                size += 1
                terms.append(x)
        
        total %= 101
        subsets_by_residue[total].append((size, terms))
        
    # Find minimum L such that all residues have a subset of size <= L
    # Max possible size is 10
    min_L = 10
    for L in range(1, 11):
        all_covered = True
        for r in range(101):
            if not subsets_by_residue[r]:
                all_covered = False
                break
            # Check if any subset has size <= L
            has_small = any(s <= L for s, _ in subsets_by_residue[r])
            if not has_small:
                all_covered = False
                break
        if all_covered:
            min_L = L
            break
            
    # Construct the result
    result = []
    for r in range(101):
        # Find the smallest subset for this residue
        # If multiple have same min size, pick any (or first one)
        best_subset = None
        best_size = min_L + 1 # Should not happen if min_L is correct
        
        for size, terms in subsets_by_residue[r]:
            if size < best_size:
                best_size = size
                best_subset = terms
                
        # Fallback if something went wrong (should not happen)
        if best_subset is None:
            # Pick any subset
            best_subset = subsets_by_residue[r][0][1]
            
        result.append(best_subset)
        
    return result
# EVOLVE-BLOCK-END
