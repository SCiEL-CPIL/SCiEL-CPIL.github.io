"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflict graph using bitmasks for fast operations
    # conflict_mask[i] is a bitmask where bit j is set if i conflicts with j
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    # Helper: check if x can be added to current_set (represented as bitmask)
    # current_mask has bit i set if i is in the set
    def can_add_mask(x, current_mask):
        # x can be added if it doesn't conflict with any element in current_mask
        # conflict_mask[x] & current_mask == 0 means no conflicts
        return (conflict_mask[x] & current_mask) == 0
    
    # Convert set to bitmask
    def set_to_mask(s):
        mask = 0
        for x in s:
            mask |= (1 << x)
        return mask
    
    # Convert bitmask to list
    def mask_to_list(mask):
        result = []
        for x in range(1, N + 1):
            if mask & (1 << x):
                result.append(x)
        return result
    
    odds = [x for x in numbers if x % 2 == 1]
    evens = [x for x in numbers if x % 2 == 0]
    
    best_size = 0
    best_mask = 0
    
    # Approach 1: Start with odds, greedily add evens
    current_mask = set_to_mask(set(odds))
    for e in evens:
        if can_add_mask(e, current_mask):
            current_mask |= (1 << e)
    if current_mask.bit_count() > best_size:
        best_size = current_mask.bit_count()
        best_mask = current_mask
    
    # Approach 2: Randomized greedy with multiple restarts
    random.seed(42)
    for trial in range(500):
        if time.time() > deadline:
            break
        order = numbers[:]
        random.shuffle(order)
        current_mask = 0
        for x in order:
            if can_add_mask(x, current_mask):
                current_mask |= (1 << x)
        sz = current_mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = current_mask
    
    # Approach 3: Local search - try to improve by swapping elements
    if time.time() < deadline:
        current_mask = best_mask
        current_size = best_size
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing one element and adding two non-conflicting ones
            # Or try single swaps
            in_list = mask_to_list(current_mask)
            not_in_list = [x for x in numbers if not (current_mask & (1 << x))]
            random.shuffle(not_in_list)
            
            # Try single swaps
            for out in in_list:
                if time.time() > deadline:
                    break
                for inp in not_in_list:
                    if inp == out:
                        continue
                    # Check if inp conflicts with out
                    if conflict_mask[inp] & (1 << out):
                        continue
                    # Check if inp conflicts with any other element in current (excluding out)
                    temp_mask = current_mask & ~(1 << out)
                    if conflict_mask[inp] & temp_mask:
                        continue
                    # Valid swap
                    current_mask = temp_mask | (1 << inp)
                    improved = True
                    break
                if improved:
                    break
            
            # Try adding a new element if possible
            if not improved:
                for inp in not_in_list:
                    if can_add_mask(inp, current_mask):
                        current_mask |= (1 << inp)
                        current_size = current_mask.bit_count()
                        improved = True
                        break
        
        if current_mask.bit_count() > best_size:
            best_size = current_mask.bit_count()
            best_mask = current_mask
    
    # Approach 4: More aggressive local search - try multiple random restarts from best
    if time.time() < deadline:
        for restart in range(200):
            if time.time() > deadline:
                break
            current_mask = best_mask
            current_size = best_mask.bit_count()
            
            # Random perturbation: remove k random elements
            k = random.randint(1, 3)
            in_list = mask_to_list(current_mask)
            if len(in_list) <= k:
                continue
            to_remove = random.sample(in_list, k)
            for x in to_remove:
                current_mask &= ~(1 << x)
            current_size = current_mask.bit_count()
            
            # Greedily add back elements
            not_in_list = [x for x in numbers if not (current_mask & (1 << x))]
            random.shuffle(not_in_list)
            for x in not_in_list:
                if can_add_mask(x, current_mask):
                    current_mask |= (1 << x)
            
            if current_mask.bit_count() > best_size:
                best_size = current_mask.bit_count()
                best_mask = current_mask
    
    # Final: try to add any remaining numbers
    current_mask = best_mask
    for x in numbers:
        if not (current_mask & (1 << x)) and can_add_mask(x, current_mask):
            current_mask |= (1 << x)
    
    best_result = mask_to_list(current_mask)
    
    # Ensure nonempty
    if not best_result:
        best_result = [1]
    
    return best_result
# EVOLVE-BLOCK-END
