import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0
    
    n = 256
    max_sum = 2 * n
    
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    valid_numbers = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_numbers.append(a)
    
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict masks
    # conflict_masks[i] has bit j set if i and j conflict
    conflict_masks = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b > a and b in val_to_idx:
                j = val_to_idx[b]
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i)
                
    # Static degrees
    static_degrees = [bin(mask).count('1') for mask in conflict_masks]
    sorted_indices = sorted(range(num_valid), key=lambda x: static_degrees[x])
    
    def add_fitting(s_mask):
        """Add all nodes that fit into the current set (represented as bitmask) in a single pass."""
        for j in range(num_valid):
            if not (s_mask & (1 << j)): # j not in s
                if not (conflict_masks[j] & s_mask): # j doesn't conflict with s
                    s_mask |= (1 << j)
        return s_mask
    
    # 1. Greedy based on static degree
    best_mask = 0
    remaining_mask = (1 << num_valid) - 1
    for v in sorted_indices:
        if (remaining_mask & (1 << v)):
            best_mask |= (1 << v)
            # Remove v and its neighbors from remaining
            remove = (1 << v) | conflict_masks[v]
            remaining_mask &= ~remove
    best_mask = add_fitting(best_mask)
    
    # 2. Random Greedy Trials
    max_trials = 300
    for trial in range(max_trials):
        if time.time() - start_time > deadline - 5:
            break
        random.seed(trial)
        order = list(range(num_valid))
        random.shuffle(order)
        s_mask = 0
        for idx in order:
            if not (s_mask & (1 << idx)):
                if not (conflict_masks[idx] & s_mask):
                    s_mask |= (1 << idx)
        s_mask = add_fitting(s_mask)
        if bin(s_mask).count('1') > bin(best_mask).count('1'):
            best_mask = s_mask
    
    # 3. Local Search (Swap-based)
    if time.time() - start_time < deadline - 10:
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            # Try to add any fitting node
            new_mask = add_fitting(best_mask)
            if new_mask != best_mask:
                best_mask = new_mask
                improved = True
                continue
            
            # Try swaps
            s_indices = [i for i in range(num_valid) if (best_mask & (1 << i))]
            random.shuffle(s_indices)
            
            for i in s_indices:
                if time.time() - start_time > deadline - 5:
                    break
                rem_mask = best_mask & ~(1 << i)
                candidate_mask = add_fitting(rem_mask)
                if bin(candidate_mask).count('1') > bin(best_mask).count('1'):
                    best_mask = candidate_mask
                    improved = True
                    break
                    
    result = [valid_numbers[i] for i in range(num_valid) if (best_mask & (1 << i))]
    if not result:
        return [1]
    return result
# EVOLVE-BLOCK-END
