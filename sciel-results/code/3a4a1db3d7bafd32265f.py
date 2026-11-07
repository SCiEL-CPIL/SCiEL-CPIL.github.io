"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    N = 256
    
    # Precompute conflict masks
    # conflict_masks[i] has bit j set if gcd(i, j) > 1
    conflict_masks = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and gcd(i, j) > 1:
                mask |= (1 << j)
        conflict_masks[i] = mask
        
    best_sum = 0
    selected = []
    
    # Memoization: map from available_mask to max current_sum seen
    visited = {}
    
    def backtrack(current_sum, available_mask):
        nonlocal best_sum, selected
        
        if time.time() > deadline:
            return
            
        # Calculate upper bound
        # Sum of all values in available_mask
        rem_sum = 0
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            rem_sum += idx + 1
            m ^= lsb
            
        if current_sum + rem_sum <= best_sum:
            return
            
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
            return
            
        # Memoization check
        if available_mask in visited:
            if visited[available_mask] >= current_sum:
                return
        else:
            visited[available_mask] = current_sum
            
        # Pick the largest available number
        # Highest set bit
        lsb = 1 << (available_mask.bit_length() - 1)
        idx = lsb.bit_length() - 1
        val = idx + 1
        
        # Option 1: Include val
        selected.append(val)
        new_mask = available_mask & ~lsb & ~conflict_masks[val]
        backtrack(current_sum + val, new_mask)
        selected.pop()
        
        # Option 2: Exclude val
        new_mask = available_mask & ~lsb
        backtrack(current_sum, new_mask)

    initial_mask = (1 << (N + 1)) - 2  # Bits 1..N set
    
    backtrack(0, initial_mask)
    
    # Ensure types are Python int
    result = [int(x) for x in selected]
    
    return result

# EVOLVE-BLOCK-END
