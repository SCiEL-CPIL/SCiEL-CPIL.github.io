"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def _is_reciprocal_sum_free(s, deadline=None):
    """Check if s is reciprocal-subsum-free."""
    s_list = list(s)
    n = len(s_list)
    
    # For each x in s, check if 1/x can be written as sum of 1/y for y in s, y != x
    for x in s_list:
        if deadline is not None and time.time() > deadline:
            return None  # Timeout, unknown result
        
        target = Fraction(1, x)
        others = [y for y in s_list if y != x]
        # DFS for subset sum
        m = len(others)
        if m == 0:
            continue
        # Sort others in ascending order of y (descending 1/y) to prune
        others_sorted = sorted(others)
        
        # Suffix sum of 1/y for pruning
        suffix_sum = [Fraction(0, 1)] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, others_sorted[i])
        
        # If total sum of all reciprocals of others is less than target, skip
        if suffix_sum[0] < target:
            continue
            
        found = [False]
        
        def dfs(idx, curr):
            if found[0]:
                return
            if deadline is not None and time.time() > deadline:
                return
            if curr == target:
                found[0] = True
                return
            if idx == m:
                return
            if curr > target:
                return
            if curr + suffix_sum[idx] < target:
                return
            
            # Option 1: include others_sorted[idx]
            new_curr = curr + Fraction(1, others_sorted[idx])
            if new_curr <= target:
                dfs(idx + 1, new_curr)
                if found[0]:
                    return
            # Option 2: exclude others_sorted[idx]
            dfs(idx + 1, curr)
        
        dfs(0, Fraction(0, 1))
        if found[0]:
            return False
    return True

def _local_search(s, deadline):
    """Try to improve set s by removing elements that cause conflicts, or adding elements."""
    s_set = set(s)
    current_size = len(s_set)
    best_set = set(s_set)
    best_size = current_size
    
    # First, remove elements until valid
    while True:
        if deadline is not None and time.time() > deadline:
            break
        is_free = _is_reciprocal_sum_free(list(best_set), deadline)
        if is_free is True:
            break
        elif is_free is None:
            # Timeout
            return best_set
        else:
            # Find a conflict and remove an element
            # For simplicity, just pick a random element to remove if we can't find specific conflict quickly
            # Or use the greedy logic
            # Let's try to find which x is the problem
            s_list = sorted(best_set)
            removed = False
            for x in s_list:
                if deadline is not None and time.time() > deadline:
                    break
                target = Fraction(1, x)
                others = [y for y in s_list if y != x]
                if not others:
                    continue
                others_sorted = sorted(others)
                m = len(others_sorted)
                suffix_sum = [Fraction(0, 1)] * (m + 1)
                for i in range(m - 1, -1, -1):
                    suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, others_sorted[i])
                if suffix_sum[0] < target:
                    continue
                found = [False]
                def dfs(idx, curr):
                    if found[0] or (deadline is not None and time.time() > deadline):
                        return
                    if curr == target:
                        found[0] = True
                        return
                    if idx == m:
                        return
                    if curr > target:
                        return
                    if curr + suffix_sum[idx] < target:
                        return
                    new_curr = curr + Fraction(1, others_sorted[idx])
                    if new_curr <= target:
                        dfs(idx + 1, new_curr)
                        if found[0]:
                            return
                    dfs(idx + 1, curr)
                dfs(0, Fraction(0, 1))
                if found[0]:
                    # Conflict found for x. Remove x? Or remove one of the others?
                    # Removing x is always safe for that x.
                    # Let's try removing x first.
                    best_set.discard(x)
                    removed = True
                    break
            if not removed:
                # No specific conflict found via this loop (maybe timeout or bug), break
                break

    # Now try to add elements
    for i in range(1, 25):
        if deadline is not None and time.time() > deadline:
            break
        if i in best_set:
            continue
        candidate = best_set | {i}
        is_free = _is_reciprocal_sum_free(list(candidate), deadline)
        if is_free is True:
            best_set = candidate
            best_size = len(best_set)
        elif is_free is None:
            break
            
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Start with a large set and reduce
    # Try different initial sets
    best_result = list(range(13, 25))
    best_size = len(best_result)
    
    # Strategy 1: Greedy add from empty
    s = set()
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        candidate = s | {i}
        if _is_reciprocal_sum_free(list(candidate), deadline):
            s = candidate
    if len(s) > best_size:
        best_result = sorted(s)
        best_size = len(s)
        
    # Strategy 2: Local search starting from high numbers
    s_start = set(range(15, 25)) # 15..24
    s_improved = _local_search(s_start, deadline)
    if len(s_improved) > best_size:
        best_result = sorted(s_improved)
        best_size = len(s_improved)
        
    # Strategy 3: Random restarts
    attempts = 0
    while time.time() < deadline - 5: # Leave 5s for final check
        attempts += 1
        # Random subset of [1, 24]
        # Start with a random set of size ~15
        rand_set = set(random.sample(range(1, 25), 15))
        s_improved = _local_search(rand_set, deadline)
        if len(s_improved) > best_size:
            best_result = sorted(s_improved)
            best_size = len(s_improved)
            
    return best_result

# EVOLVE-BLOCK-END
