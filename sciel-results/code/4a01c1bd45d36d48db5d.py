"""Optimized construct: minimum-cardinality assignment per residue via size-ordered search."""
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    p = 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
    
    denominators = list(range(1, 11))
    
    # Precompute residue for every subset (by mask)
    # mask: bits 0..9 correspond to denominators 1..10
    subset_residue = [0] * (1 << 10)
    subset_size = [0] * (1 << 10)
    for mask in range(1, 1 << 10):
        # Use lowest set bit to build up
        lb = mask & (-mask)
        idx = lb.bit_length() - 1  # 0-indexed, corresponds to denominator idx+1
        prev = mask ^ lb
        subset_residue[mask] = (subset_residue[prev] + inv[idx + 1]) % p
        subset_size[mask] = subset_size[prev] + 1
    
    # Group masks by size
    by_size = [[] for _ in range(11)]
    for mask in range(1, 1 << 10):
        by_size[subset_size[mask]].append(mask)
    
    # Assign minimum-cardinality subset to each residue
    assigned = [None] * p  # assigned[r] = list of denominators
    for k in range(1, 11):
        if time.time() > deadline:
            break
        for mask in by_size[k]:
            r = subset_residue[mask]
            if assigned[r] is None:
                # Extract denominators from mask
                terms = [i + 1 for i in range(10) if mask >> i & 1]
                assigned[r] = terms
    
    # If any residue is still unassigned (shouldn't happen, but safety), fill with full set
    full_set = list(range(1, 11))
    for r in range(p):
        if assigned[r] is None:
            assigned[r] = full_set
    
    return [assigned[r] for r in range(p)]
