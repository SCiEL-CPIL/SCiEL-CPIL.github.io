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
    
    # sums_by_size[k] = set of residues achievable with exactly k elements
    # masks_by_size_sum[(k, r)] = list of masks
    sums_by_size = [set() for _ in range(N + 1)]
    masks_by_size_sum = {}
    
    # Precompute all subsets
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
            
    # Find minimum T such that all residues are covered by sizes 1..T
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

    # Assign each residue to its minimum size <= min_T
    result = []
    for r in range(P):
        final_mask = None
        for k in range(1, min_T + 1):
            if r in sums_by_size[k]:
                final_mask = masks_by_size_sum[(k, r)][0]
                break
        
        terms = []
        if final_mask is not None:
            for i in range(1, N + 1):
                if final_mask & (1 << (i - 1)):
                    terms.append(i)
        result.append(terms)
        
    return result
# EVOLVE-BLOCK-END
