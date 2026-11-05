import time
from math import lcm

def _can_form_target_int(target_int, values_int, n_values):
    if n_values == 0:
        return False
    total = sum(values_int)
    if total < target_int:
        return False
    if target_int <= 0:
        return False
    
    # Filter values
    filtered = [v for v in values_int if v <= target_int]
    if not filtered:
        return False
        
    n = len(filtered)
    if n == 0:
        return False
        
    # Check single
    for v in filtered:
        if v == target_int:
            return True

    half = n // 2
    left = filtered[:half]
    right = filtered[half:]
    
    left_sums = {0}
    for v in left:
        new_sums = set()
        for s in left_sums:
            ns = s + v
            if ns <= target_int:
                new_sums.add(ns)
        left_sums.update(new_sums)
        
    if target_int in left_sums:
        return True
        
    right_sums = {0}
    for v in right:
        new_sums = set()
        for s in right_sums:
            ns = s + v
            if ns <= target_int:
                new_sums.add(ns)
        right_sums.update(new_sums)
        
    for sr in right_sums:
        need = target_int - sr
        if need in left_sums:
            return True
            
    return False

def _is_valid_set(S):
    S = sorted(S)
    n = len(S)
    lcm_val = 1
    for x in S:
        lcm_val = lcm(lcm_val, x)
        
    for idx, x in enumerate(S):
        target_int = lcm_val // x
        values_int = [lcm_val // y for i, y in enumerate(S) if i != idx]
        if _can_form_target_int(target_int, values_int, n - 1):
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 85
    
    # Start with {1, 12, ..., 24}
    best = {1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
    if not _is_valid_set(best):
        best = set(range(13, 25))
        
    best_size = len(best)
    best_sum = sum(best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        added = False
        for c in range(1, 25):
            if c in best:
                continue
            trial = best | {c}
            if _is_valid_set(trial):
                best = trial
                best_size = len(best)
                best_sum = sum(best)
                improved = True
                added = True
                break
        
        if added:
            continue
        
        # Try swap to improve sum (heuristic for safety)
        for c in range(1, 25):
            if c in best:
                continue
            for d in best:
                if time.time() > deadline:
                    break
                trial = best - {d} | {c}
                if _is_valid_set(trial):
                    if sum(trial) > best_sum:
                        best = trial
                        best_sum = sum(best)
                        improved = True
                        break
            if improved:
                break
                
    return sorted(best)
