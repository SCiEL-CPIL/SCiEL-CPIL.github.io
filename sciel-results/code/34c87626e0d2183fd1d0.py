import time
from math import gcd

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We target 140s.
    deadline = start_time + 140.0
    
    N = 511
    # Weights: w[a] = 1 / (512 - a)
    # Index a from 1 to 511
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
        
    # Conflict Masks
    # conflict_mask[a] has bit (b-1) set if gcd(a, b) > 1
    # Note: a conflicts with itself, so bit (a-1) is set.
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if gcd(a, b) > 1:
                mask |= (1 << (b - 1))
        conflict_mask[a] = mask
        
    best_score = 0.0
    best_set = []
    current_set = []
    
    def sum_weights(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx + 1]
            m ^= lsb
        return s

    def search(available_mask, current_score):
        nonlocal best_score, best_set, current_set
        
        # Time check
        if time.time() > deadline:
            return
            
        if available_mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
            return
            
        # Upper Bound
        # If current score + max possible from remaining <= best, prune
        # We use the sum of all remaining weights as the bound
        ub = current_score + sum_weights(available_mask)
        if ub <= best_score:
            return
            
        # Find the largest a in available_mask
        # bit_length() returns the number of bits necessary to represent the integer.
        # For mask ...1000, bit_length is 4. The index of the highest bit is 3.
        # The value a is index + 1.
        # So a = bit_length().
        a_val = available_mask.bit_length()
        
        # Branch 1: Include a_val
        # It is guaranteed to be in available_mask because it's the highest bit.
        current_set.append(a_val)
        # Remove a_val and all its conflicts
        new_mask = available_mask & ~conflict_mask[a_val]
        search(new_mask, current_score + weights[a_val])
        current_set.pop()
        
        # Branch 2: Exclude a_val
        # Remove only a_val
        new_mask = available_mask & ~(1 << (a_val - 1))
        search(new_mask, current_score)

    # Initial mask: all bits 0 to 510 set
    initial_mask = (1 << N) - 1
    search(initial_mask, 0.0)
    
    return best_set
