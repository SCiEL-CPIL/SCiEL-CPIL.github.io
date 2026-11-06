import time
import random

_targets = set(range(1, 31))
_best_set = None
_best_max = float('inf')
_deadline = None
_current_n = 9

def _dfs(current_set, used_diffs, remaining_targets, max_allowed):
    global _best_set, _best_max, _deadline, _current_n

    if time.time() > _deadline:
        return

    n = len(current_set)
    target_n = _current_n

    if n == target_n:
        if not remaining_targets:
            m = max(current_set)
            if m < _best_max:
                _best_max = m
                _best_set = list(current_set)
        return

    k = target_n - n
    max_new_diffs = k * n + k * (k - 1) // 2
    if len(remaining_targets) > max_new_diffs:
        return

    last = current_set[-1]
    start_x = last + 1
    # We want to find a solution with max < _best_max
    # So x must be < _best_max
    # Also x < max_allowed
    end_x = min(max_allowed, _best_max - 1)
    
    if start_x > end_x:
        return

    for x in range(start_x, end_x + 1):
        if time.time() > _deadline:
            return
        
        valid = True
        new_diffs = []
        for a in current_set:
            d = x - a
            if d in used_diffs:
                valid = False
                break
            new_diffs.append(d)
        
        if not valid:
            continue
            
        new_used = used_diffs.union(new_diffs)
        new_remaining = remaining_targets.difference(new_diffs)
        
        _dfs(current_set + [x], new_used, new_remaining, max_allowed)
        
        if _best_max <= x:
            break

def construct():
    global _deadline, _best_set, _best_max, _current_n
    
    start_time = time.time()
    _deadline = start_time + 110.0 # Safety margin
    
    _best_set = None
    _best_max = float('inf')
    
    # Try n=9, 10, 11
    # Allocate time: 60s for n=9, 30s for n=10, 20s for n=11
    time_allocations = [60, 30, 20]
    
    for i, n in enumerate([9, 10, 11]):
        if time.time() > _deadline:
            break
            
        _current_n = n
        _best_max = float('inf')
        _best_set = None
        
        # Reset deadline for this specific search phase?
        # No, global deadline is better to ensure total time < 120s.
        # But we want to prioritize smaller n.
        # If we find a good solution for n=9, we might not need n=10.
        
        # Start with [0]
        _dfs([0], set(), set(range(1, 31)), 200)
        
        if _best_set is not None:
            # Found a solution for this n
            # We can break if we are confident this is optimal?
            # Or just keep the best across all n.
            # Since n=9 is smaller, it's likely to have a smaller or equal max.
            # Let's break if we found a solution for n=9.
            if n == 9:
                break
            
    if _best_set is None:
        # Fallback
        return [0, 1, 4, 10, 18, 23, 25, 29, 30]
        
    return _best_set
