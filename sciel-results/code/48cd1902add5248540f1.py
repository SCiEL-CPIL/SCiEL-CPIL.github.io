"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _START

def _best_value(rows):
    return max(len(r) for r in rows)

def construct():
    MOD = 101
    N = 10
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)
        
    # best_subset[r] = (size, mask)
    # We want the minimum size for each residue.
    # Initialize with a large size
    best_size = [N + 1] * MOD
    best_mask = [0] * MOD
    
    # Iterate over all non-empty subsets
    # We can iterate by mask from 1 to 2^N - 1
    # To ensure we find the smallest size first, we can just check if current size < best_size
    # But iterating by popcount is cleaner for early termination if we wanted, 
    # but here we just need the global min per residue.
    
    # Precompute all subset sums and sizes for speed
    # There are 2^10 - 1 = 1023 subsets, which is very small
    
    # Let's precompute for each mask: (sum_mod, size)
    subset_info = {}
    for mask in range(1, 1 << N):
        s = 0
        m = mask
        size = 0
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            s += inv[x]
            size += 1
            m -= lsb
        subset_info[mask] = (s % MOD, size)
    
    # Now iterate through all masks to find best for each residue
    for mask in range(1, 1 << N):
        r, size = subset_info[mask]
        if size < best_size[r]:
            best_size[r] = size
            best_mask[r] = mask
            
    # Check if all residues are covered
    all_covered = all(best_mask[r] != 0 for r in range(MOD))
    
    if not all_covered:
        # Fallback: use a different strategy
        # Try to cover remaining residues with larger subsets
        uncovered = [r for r in range(MOD) if best_mask[r] == 0]
        for r in uncovered:
            # Find any subset that sums to r
            for mask in range(1, 1 << N):
                sr, _ = subset_info[mask]
                if sr == r:
                    best_mask[r] = mask
                    best_size[r] = _
                    break
    
    # Construct rows
    rows = []
    for r in range(MOD):
        if best_mask[r] == 0:
            # Fallback, should not happen
            rows.append([1])
        else:
            mask = best_mask[r]
            terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            rows.append(terms)
            
    return rows

# EVOLVE-BLOCK-END
