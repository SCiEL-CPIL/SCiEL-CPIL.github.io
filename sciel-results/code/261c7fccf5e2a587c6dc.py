"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256, 'maximum_size': 30}

def _check_divisibility_conflict(existing_sums, new_sums):
    """
    Check if any pair (a from existing_sums, b from new_sums) has a|b or b|a,
    or if any pair within new_sums has divisibility, or within existing_sums.
    Since existing_sums is already valid, we only need to check:
    - existing vs new
    - new vs new
    """
    for b in new_sums:
        for a in existing_sums:
            if a == b:
                return False  # collision
            if a < b:
                if b % a == 0:
                    return False
            else:
                if a % b == 0:
                    return False
    # Check new vs new
    new_list = sorted(new_sums)
    for i in range(len(new_list)):
        for j in range(i + 1, len(new_list)):
            if new_list[j] % new_list[i] == 0:
                return False
    return True

def _try_add_element(sums_set, x):
    """
    Try to add element x to a set whose nonempty subset sums are sums_set.
    Returns (valid, new_sums_set) or (False, None).
    """
    new_sums = set()
    for s in sums_set:
        ns = s + x
        new_sums.add(ns)
    new_sums.add(x)
    
    # Check for collisions: new_sums must not overlap with sums_set
    if new_sums & sums_set:
        return False, None
    
    # Check divisibility
    if not _check_divisibility_conflict(sums_set, new_sums):
        return False, None
    
    new_total = sums_set | new_sums
    return True, new_total

def construct():
    deadline = time.time() + 80  # 80 second deadline, leaving 10s safety margin
    
    best_set = []
    best_size = 0
    
    def dfs(current_set, sums_set, candidates):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        if not candidates:
            return
        
        for i in range(len(candidates)):
            x = candidates[i]
            remaining = candidates[:i] + candidates[i+1:]
            
            valid, new_sums = _try_add_element(sums_set, x)
            if valid:
                dfs(current_set + [x], new_sums, remaining)
    
    # Try multiple strategies
    base_candidates = list(range(1, 257))
    
    strategies = [
        lambda: list(range(256, 0, -1)),  # descending
        lambda: list(range(1, 257)),       # ascending
        lambda: list(range(128, 257)),     # upper half
        lambda: list(range(64, 257)),      # upper 2/3
        lambda: list(range(1, 129)),       # lower half
    ]
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        strat_idx = r % len(strategies)
        candidates = strategies[strat_idx]()
        
        if r % 2 == 1:
            random.shuffle(candidates)
        
        dfs([], set(), candidates)
    
    return best_set if best_set else [31, 32, 34]

# EVOLVE-BLOCK-END
