import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    required = [1, 2, 5, 11]
    max_val = 284
    
    # Precompute all pairs sums for required
    req_sums = set()
    for i in range(len(required)):
        for j in range(i, len(required)):
            req_sums.add(required[i] + required[j])
    
    # Bitmask for sums
    init_mask = 0
    for s in req_sums:
        init_mask |= (1 << s)
        
    base_candidates = [x for x in range(1, max_val + 1) if x not in required]
    
    best_set = required[:]
    best_size = len(required)
    
    def dfs(mask, cand_idx, curr_set, target_size, cands):
        if len(curr_set) == target_size:
            return True
        if cand_idx == len(cands):
            return False
            
        # Pruning: not enough candidates left to reach target
        if len(curr_set) + (len(cands) - cand_idx) < target_size:
            return False
            
        # Time check
        if time.time() > deadline:
            return False
            
        for i in range(cand_idx, len(cands)):
            c = cands[i]
            
            # Check if 2*c is already in mask
            if mask & (1 << (2*c)):
                continue
                
            # Check if c + y is in mask for any y in curr_set
            valid = True
            for y in curr_set:
                if mask & (1 << (c + y)):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # If valid, add c
            new_mask = mask | (1 << (2*c))
            for y in curr_set:
                new_mask |= (1 << (c + y))
                
            curr_set.append(c)
            
            if dfs(new_mask, i + 1, curr_set, target_size, cands):
                return True
                
            curr_set.pop()
            
        return False

    # Generate multiple orderings to escape local minima
    orders = [base_candidates[:], base_candidates[::-1]]
    for _ in range(20):
        shuffled = base_candidates[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    # Try to find a set of maximum size
    for target_size in range(20, len(required) + 1, -1):
        found = False
        for order in orders:
            if time.time() > deadline:
                break
                
            curr_set = required[:]
            if dfs(init_mask, 0, curr_set, target_size, order):
                best_set = curr_set[:]
                best_size = target_size
                found = True
                break
        
        if found:
            break
            
    return sorted(best_set)
