import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))

    # Precompute modular inverses
    inverses = [0] * MAX_DENOM
    for i, d in enumerate(denominators):
        inverses[i] = pow(d, -1, MOD)

    max_mask = 1 << MAX_DENOM

    # Precompute sum and length for all masks using DP
    mask_sum = [0] * max_mask
    mask_len = [0] * max_mask
    for mask in range(1, max_mask):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        mask_sum[mask] = (mask_sum[prev] + inverses[idx]) % MOD
        mask_len[mask] = mask_len[prev] + 1

    # For each residue, collect all masks that produce it, sorted by length
    masks_by_residue = defaultdict(list)
    for mask in range(1, max_mask):
        s = mask_sum[mask]
        masks_by_residue[s].append(mask)
    
    # Sort each residue's masks by length (shorter first)
    for r in masks_by_residue:
        masks_by_residue[r].sort(key=lambda m: mask_len[m])

    # Try to find an assignment minimizing the maximum row length
    # We'll use a greedy approach with backtracking for small cases
    
    # First, let's see what the minimum possible max length is
    # Try max_len = 1, 2, 3, ...
    
    best_assignment = None
    best_max_len = MAX_DENOM + 1
    
    for max_len in range(1, MAX_DENOM + 1):
        if time.time() > deadline - 10:
            break
            
        # Try to assign masks to residues using only masks of length <= max_len
        # This is a bipartite matching problem: residues on one side, masks on the other
        # But multiple residues can use the same mask (denominators may recur across rows)
        
        # Actually, the constraint is: each row must have distinct denominators, but 
        # the same denominator can appear in different rows. So each residue just needs
        # ONE valid mask. There's no global constraint on mask usage.
        
        # So for each residue, we just need to check if there exists ANY mask of length <= max_len
        # that produces that residue.
        
        can_assign = True
        assignment = [None] * MOD
        for r in range(MOD):
            if r not in masks_by_residue:
                can_assign = False
                break
            found = False
            for m in masks_by_residue[r]:
                if mask_len[m] <= max_len:
                    assignment[r] = m
                    found = True
                    break
            if not found:
                can_assign = False
                break
        
        if can_assign:
            best_assignment = assignment
            best_max_len = max_len
            break
    
    # If no assignment found (shouldn't happen), use the greedy fallback
    if best_assignment is None:
        best_assignment = [None] * MOD
        for r in range(MOD):
            if r in masks_by_residue and masks_by_residue[r]:
                # Pick the shortest mask
                best_assignment[r] = masks_by_residue[r][0]
            else:
                best_assignment[r] = 1  # Fallback
        best_max_len = MAX_DENOM + 1

    # Construct the final result
    results = [None] * MOD
    for r in range(MOD):
        mask = best_assignment[r]
        if mask is None:
            mask = 1
        combo = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            combo.append(denominators[idx])
            m ^= lsb
        results[r] = combo

    return results
