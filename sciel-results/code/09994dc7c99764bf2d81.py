import time
import sys
import random
from math import gcd

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    
    # Precompute compatibility for all pairs
    # compat[i] is a bitmask of vertices j such that lcm(vertices[i], vertices[j]) > N
    compat = [0] * n
    for i in range(n):
        a = vertices[i]
        for j in range(i + 1, n):
            b = vertices[j]
            g = gcd(a, b)
            if a * b // g > N:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
                
    weights = [1.0 / v for v in vertices]
    
    # Split into Core (small, high weight) and Remainder (large, low weight)
    # Core: indices 0 to K-1 corresponding to values 2 to K+1
    # Let's choose K such that 2^K is manageable.
    # Values 2..16 are indices 0..14. 2^15 = 32768 subsets.
    # Values 2..32 are indices 0..30. 2^31 is too large.
    # Let's stick to 2..16 for core enumeration.
    core_max_val = 16
    core_indices = [i for i, v in enumerate(vertices) if v <= core_max_val]
    core_size = len(core_indices)
    
    # Precompute weights for core and remainder
    # We will iterate over all subsets of core_indices
    
    best_val = 0.0
    best_mask = 0
    
    # Helper to calculate value of a mask
    def get_value(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            s += weights[idx]
            m ^= lsb
        return s

    # Iterate over all subsets of the core
    # We represent the core subset by a bitmask over core_indices
    # But we need to map back to global indices for compatibility checks
    
    # For speed, precompute for each core subset:
    # 1. Is it a valid independent set within the core?
    # 2. What is the set of remainder indices that are compatible with this core subset?
    
    # Instead of precomputing everything, we do it on the fly but efficiently.
    
    # Optimization:
    # For each core subset, we can maintain a "blocked" bitmask for the remainder.
    # A remainder index j is blocked if it is incompatible with ANY core index i in the subset.
    # blocked[j] = OR over i in subset of (~compat[i] & remainder_mask)
    
    remainder_mask = 0
    for i in range(n):
        if vertices[i] > core_max_val:
            remainder_mask |= (1 << i)
            
    # Precompute incompatibility masks for core indices relative to remainder
    # inc_core_to_rem[i] = bitmask of remainder indices j such that NOT compat[i][j]
    inc_core_to_rem = [0] * core_size
    for ci, i in enumerate(core_indices):
        mask = 0
        for j in range(n):
            if vertices[j] > core_max_val:
                if not (compat[i] & (1 << j)):
                    mask |= (1 << j)
        inc_core_to_rem[ci] = mask

    # Iterate through all 2^core_size subsets
    for mask_core in range(1 << core_size):
        # Check time
        if time.time() > deadline:
            break
            
        # Check if mask_core is a valid independent set in the core
        valid = True
        m = mask_core
        while m:
            lsb = m & -m
            ci = (lsb.bit_length() - 1)
            i = core_indices[ci]
            # Check if i is compatible with all other bits in mask_core
            # If any other bit k in mask_core is not compatible with i, invalid
            # We can precompute core_compat_mask[ci] = bitmask of other core indices compatible with i
            # But let's just check against the rest of the mask
            rest = mask_core ^ lsb
            # If rest has any bit that is incompatible with i, invalid
            # We need a quick check.
            # Let's precompute core_incompat[ci] = bitmask of core indices (local to core) incompatible with i
            pass # This inner check is O(core_size) per subset, which is 15 * 32768 ~ 500k ops. Fine.
            
            # Actually, let's precompute core_incompat for speed
            # This needs to be outside the loop.
            
        # Let's restructure to precompute core_incompat
        # core_incompat[ci] is a bitmask of core indices (0..core_size-1) that are NOT compatible with core_indices[ci]
        
    # Precompute core_incompat
    core_incompat = [0] * core_size
    for ci in range(core_size):
        i = core_indices[ci]
        mask = 0
        for cj in range(core_size):
            if ci == cj:
                continue
            j = core_indices[cj]
            if not (compat[i] & (1 << j)):
                mask |= (1 << cj)
        core_incompat[ci] = mask

    # Now iterate subsets
    for mask_core in range(1 << core_size):
        if time.time() > deadline:
            break
            
        # Check validity of core subset
        valid = True
        m = mask_core
        while m:
            lsb = m & -m
            ci = (lsb.bit_length() - 1)
            # If any incompatible core index is present in mask_core, invalid
            if mask_core & core_incompat[ci]:
                valid = False
                break
            m ^= lsb
            
        if not valid:
            continue
            
        # Calculate blocked remainder
        blocked_rem = 0
        m = mask_core
        while m:
            lsb = m & -m
            ci = (lsb.bit_length() - 1)
            blocked_rem |= inc_core_to_rem[ci]
            m ^= lsb
            
        # Available remainder = remainder_mask & ~blocked_rem
        avail_rem = remainder_mask & ~blocked_rem
        
        # Now, we need to select a subset of avail_rem that is an independent set.
        # We want to maximize sum of weights.
        # Greedy: Sort avail_rem indices by weight descending (value ascending) and add if compatible with already selected.
        # Since we are maximizing sum, and weights are 1/v, smaller v is better.
        # So we iterate through vertices in ascending order of value.
        
        # To make this fast, we can just iterate through all remainder indices in order of value (which is index order for remainder)
        # and check if they are in avail_rem and compatible with current selection.
        
        sel_rem = 0
        val_rem = 0.0
        
        # Iterate remainder indices in order of increasing value (decreasing weight)
        # Remainder indices are those with vertices[i] > core_max_val
        # Since vertices are sorted 2..256, the remainder indices are contiguous at the end of the list?
        # No, vertices are 2..256.
        # core_max_val = 16.
        # Indices 0..14 are 2..16.
        # Indices 15..254 are 17..256.
        # So we can just iterate i from 15 to 254.
        
        for i in range(core_size, n):
            if not (avail_rem & (1 << i)):
                continue
            # Check if i is compatible with all selected in sel_rem
            # If (sel_rem & ~compat[i]) == 0, then compatible
            if (sel_rem & ~compat[i]) == 0:
                sel_rem |= (1 << i)
                val_rem += weights[i]
                
        # Total value
        # Value of core
        val_core = 0.0
        m = mask_core
        while m:
            lsb = m & -m
            ci = (lsb.bit_length() - 1)
            i = core_indices[ci]
            val_core += weights[i]
            m ^= lsb
            
        total_val = val_core + val_rem
        
        if total_val > best_val:
            best_val = total_val
            # Construct best_mask
            best_mask = sel_rem
            m = mask_core
            while m:
                lsb = m & -m
                ci = (lsb.bit_length() - 1)
                i = core_indices[ci]
                best_mask |= (1 << i)
                m ^= lsb

    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = (lsb.bit_length() - 1)
        result.append(vertices[idx])
        m ^= lsb
        
    if not result:
        return [2]
        
    return result
