"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 58.0
    
    P = 101
    N = 10
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, P - 2, P)
    
    # Map from (sum, size) to list of masks achieving it
    # size can be 1..10, sum 0..100
    # We want for each residue r, the minimum size k and a mask of size k with sum ≡ r
    
    # First pass: for each size k, compute all sums achievable
    # sums_by_size[k] = set of residues achievable with exactly k elements
    # masks_by_size_sum[(k, r)] = list of masks
    
    sums_by_size = [set() for _ in range(N + 1)]
    masks_by_size_sum = {}  # (k, r) -> [mask, ...]
    
    for k in range(1, N + 1):
        for combo in combinations(range(1, N + 1), k):
            s = 0
            mask = 0
            for i in combo:
                s += inv[i]
                mask |= (1 << (i - 1))
            r = s % P
            sums_by_size[k].add(r)
            key = (k, r)
            if key not in masks_by_size_sum:
                masks_by_size_sum[key] = []
            masks_by_size_sum[key].append(mask)
    
    # Now find minimum size for each residue
    best_size = [N + 1] * P
    best_mask = [None] * P
    
    for r in range(P):
        for k in range(1, N + 1):
            if r in sums_by_size[k]:
                best_size[r] = k
                best_mask[r] = masks_by_size_sum[(k, r)][0]
                break
    
    # Compute max row length
    max_len = max(best_size)
    
    # Try to reduce max_len using a more careful assignment
    # The above greedy approach might not be optimal in terms of max length
    # Let's check: is there an assignment where every residue gets size <= T for some T < max_len?
    # For a given T, check if every residue r has some size k <= T such that r in sums_by_size[k]
    # If yes, we can assign each residue to its minimum feasible size <= T
    
    # Find minimum T such that all residues are coverable with size <= T
    min_T = N + 1
    for T in range(1, N + 1):
        all_covered = True
        for r in range(P):
            covered = False
            for k in range(1, T + 1):
                if r in sums_by_size[k]:
                    covered = True
                    break
            if not covered:
                all_covered = False
                break
        if all_covered:
            min_T = T
            break
    
    # Assign each residue to its minimum size (which is <= min_T)
    final_size = [0] * P
    final_mask = [None] * P
    for r in range(P):
        for k in range(1, min_T + 1):
            if r in sums_by_size[k]:
                final_size[r] = k
                final_mask[r] = masks_by_size_sum[(k, r)][0]
                break
    
    # Build result
    result = []
    for r in range(P):
        m = final_mask[r]
        terms = []
        for i in range(1, N + 1):
            if m & (1 << (i - 1)):
                terms.append(i)
        result.append(terms)
    
    return result
# EVOLVE-BLOCK-END
