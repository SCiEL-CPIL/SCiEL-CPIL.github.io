"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    
    p = 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # Precompute all 2^10 subset sums using bitmask enumeration
    # mask bit i (0-indexed) corresponds to denominator (i+1)
    subset_sum = [0] * (1 << 10)
    
    for mask in range(1, 1 << 10):
        # Use lowest set bit to build incrementally
        lb = mask & (-mask)
        idx = lb.bit_length() - 1  # 0-indexed, so denominator = idx + 1
        prev = mask ^ lb
        subset_sum[mask] = (subset_sum[prev] + inv[idx + 1]) % p
    
    # For each residue, find the minimum-size subset that achieves it
    min_len = [11] * p
    best_subsets = [None] * p
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << 10)
    for mask in range(1, 1 << 10):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Group masks by their popcount
    by_size = [[] for _ in range(11)]
    for mask in range(1, 1 << 10):
        by_size[popcount[mask]].append(mask)
    
    # Precompute denominator lists for all masks
    mask_to_denoms = [None] * (1 << 10)
    for mask in range(1, 1 << 10):
        denoms = []
        for i in range(10):
            if mask & (1 << i):
                denoms.append(i + 1)
        mask_to_denoms[mask] = denoms
    
    for size in range(1, 11):
        if time.time() > deadline:
            break
        
        # Check if all residues are covered
        if all(ml < 11 for ml in min_len):
            break
        
        for mask in by_size[size]:
            s = subset_sum[mask]
            if size < min_len[s]:
                min_len[s] = size
                best_subsets[s] = mask_to_denoms[mask]
    
    # Fallback for any uncovered residues (shouldn't happen, but safety)
    for r in range(p):
        if best_subsets[r] is None:
            for mask in range(1, 1 << 10):
                if subset_sum[mask] == r:
                    best_subsets[r] = mask_to_denoms[mask]
                    min_len[r] = popcount[mask]
                    break
    
    return [best_subsets[r] for r in range(p)]
# EVOLVE-BLOCK-END
