"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
import time

def construct():
    """
    Find a large Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses randomized backtracking with bitset-based conflict checking.
    """
    required = [1, 2, 5, 11]
    N = 284
    target_size = 20
    
    def get_initial_state():
        S = sorted(required)
        used_sums = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                used_sums |= (1 << (S[i] + S[j]))
        return S, used_sums
    
    def can_add(S, used_sums, x):
        if (used_sums >> (2 * x)) & 1:
            return False
        for s in S:
            if (used_sums >> (x + s)) & 1:
                return False
        return True
    
    def add_to_state(S, used_sums, x):
        new_used_sums = used_sums | (1 << (2 * x))
        for s in S:
            new_used_sums |= (1 << (x + s))
        return S + [x], new_used_sums

    best_set = None
    best_size = 0
    start_time = time.time()
    time_limit = 110.0
    
    def dfs(S, used_sums, remaining_candidates):
        nonlocal best_set, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return False
        
        if len(S) > best_size:
            best_set = list(S)
            best_size = len(S)
            
        if best_size >= target_size:
            return True
            
        if not remaining_candidates:
            return False
            
        compatible = [x for x in remaining_candidates if can_add(S, used_sums, x)]
        
        if not compatible:
            return False
            
        random.shuffle(compatible)
        
        for x in compatible:
            new_S, new_used_sums = add_to_state(S, used_sums, x)
            new_remaining = [y for y in remaining_candidates if y != x]
            
            if dfs(new_S, new_used_sums, new_remaining):
                return True
                
        return False

    S0, used_sums0 = get_initial_state()
    initial_candidates = [x for x in range(1, N + 1) if x not in required]
    
    found_optimal = dfs(S0, used_sums0, initial_candidates)
    
    if not found_optimal and best_size < target_size:
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            random.seed()
            dfs(S0, used_sums0, initial_candidates)
            if best_size >= target_size:
                break

    result = sorted(best_set) if best_set else sorted(required)
    
    sums = set()
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
            
    if not valid or not all(x in result for x in required):
        result = sorted(required)
        
    return result

# EVOLVE-BLOCK-END
