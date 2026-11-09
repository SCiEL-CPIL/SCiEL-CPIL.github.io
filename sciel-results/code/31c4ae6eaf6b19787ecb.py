"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd

def construct():
    N = 256
    
    # Precompute coprime masks
    # mask[i] has bit j set if gcd(i, j) == 1
    coprime_masks = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if gcd(i, j) == 1:
                mask |= (1 << j)
        coprime_masks[i] = mask
        
    # Precompute lower masks: lower_mask[i] has bits 1..i set
    lower_masks = [0] * (N + 1)
    current_mask = 0
    for i in range(1, N + 1):
        current_mask |= (1 << i)
        lower_masks[i] = current_mask
        
    # Precompute sum table for 8-bit chunks
    # sum_table[chunk] = sum of indices of set bits in chunk (0-7)
    sum_table = [0] * 256
    for chunk in range(256):
        s = 0
        for b in range(8):
            if chunk & (1 << b):
                s += b
        sum_table[chunk] = s

    def sum_mask(mask):
        """Sum of indices of set bits in mask."""
        total = 0
        for k in range(0, 256, 8):
            chunk = (mask >> k) & 0xFF
            if chunk:
                total += sum_table[chunk] + k * chunk.bit_count()
        return total

    best_sum = 0
    best_selection = []
    
    # Recursive Branch and Bound
    # i: current number to consider (descending)
    # current_sum: sum of selected numbers so far
    # available_mask: bitmask of numbers coprime to all selected numbers
    
    def search(i, current_sum, available_mask, selected):
        nonlocal best_sum, best_selection
        
        if i == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = selected[:]
            return
            
        # Upper bound: current_sum + sum of all available numbers <= i
        relevant_mask = available_mask & lower_masks[i]
        if relevant_mask:
            ub = current_sum + sum_mask(relevant_mask)
        else:
            ub = current_sum
            
        if ub <= best_sum:
            return
        
        # Option 1: Pick i if possible
        if available_mask & (1 << i):
            # Pick i
            new_mask = available_mask & coprime_masks[i]
            selected.append(i)
            search(i - 1, current_sum + i, new_mask, selected)
            selected.pop()
            
        # Option 2: Skip i
        search(i - 1, current_sum, available_mask, selected)

    # Initial mask: all numbers 1..256 are available
    initial_mask = lower_masks[N]
    search(N, 0, initial_mask, [])
    
    return best_selection
# EVOLVE-BLOCK-END
