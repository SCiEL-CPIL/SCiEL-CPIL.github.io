"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def _check_valid(s, deadline=None):
    """Check if set s is reciprocal-subsum-free using integer LCM method.
    Returns True if valid, False if invalid, None on timeout."""
    s_list = sorted(s)
    n = len(s_list)
    for i, x in enumerate(s_list):
        if deadline is not None and time.time() > deadline:
            return None
        others = [y for y in s_list if y != x]
        if not others:
            continue
        
        # Compute LCM of x and all others
        L = x
        for y in others:
            L = L * y // gcd(L, y)
        
        target = L // x
        values = [L // y for y in others]
        
        if sum(values) < target:
            continue
        
        values_sorted = sorted(values, reverse=True)
        m = len(values_sorted)
        
        # Suffix sums for pruning
        suffix = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] + values_sorted[i]
        
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
            if curr + suffix[idx] < target:
                return
            # Include
            dfs(idx + 1, curr + values_sorted[idx])
            if found[0]:
                return
            # Exclude
            dfs(idx + 1, curr)
        
        dfs(0, 0)
        if found[0]:
            return False
    return True

def _find_conflict(s, deadline=None):
    """Find a specific conflict (x, subset) in set s.
    Returns (x, subset) or (None, None) if no conflict or timeout."""
    s_list = sorted(s)
    for x in s_list:
        if deadline is not None and time.time() > deadline:
            return (None, None)
        others = [y for y in s_list if y != x]
        if not others:
            continue
        
        L = x
        for y in others:
            L = L * y // gcd(L, y)
        
        target = L // x
        values = [L // y for y in others]
        
        if sum(values) < target:
            continue
        
        # Map values back to original elements for reconstruction
        val_to_elem = {}
        for y in others:
            v = L // y
            val_to_elem[v] = y
        
        values_sorted = sorted(values, reverse=True)
        m = len(values_sorted)
        
        suffix = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] + values_sorted[i]
        
        found = [False]
        path = [False] * m
        
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
            if curr + suffix[idx] < target:
                return
            path[idx] = True
            dfs(idx + 1, curr + values_sorted[idx])
            if found[0]:
                return
            path[idx] = False
            dfs(idx + 1, curr)
        
        dfs(0, 0)
        if found[0]:
            subset = [val_to_elem[values_sorted[i]] for i in range(m) if path[i]]
            return (x, subset)
    return (None, None)

def _greedy_build(order, deadline):
    """Build a valid set by greedily adding elements in the given order."""
    s = set()
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        candidate = s | {x}
        result = _check_valid(candidate, deadline)
        if result is True:
            s = candidate
        elif result is None:
            break
    return s

def _local_search(s, deadline):
    """Improve set s by removing conflicting elements and adding new ones."""
    s = set(s)
    
    # Remove conflicting elements until valid
    max_removals = 20
    for _ in range(max_removals):
        if deadline is not None and time.time() > deadline:
            break
        result = _check_valid(s, deadline)
        if result is True:
            break
        elif result is None:
            break
        else:
            # Find a conflict and remove the target element x
            x, subset = _find_conflict(s, deadline)
            if x is None:
                break
            s.discard(x)
    
    # Try to add elements
    for x in range(1, 25):
        if deadline is not None and time.time() > deadline:
            break
        if x in s:
            continue
        candidate = s | {x}
        result = _check_valid(candidate, deadline)
        if result is True:
            s = candidate
        elif result is None:
            break
    
    return s

def _backtracking_max_subset(deadline):
    """Systematic backtracking search for largest valid subset using forward checking."""
    best_size = 0
    best_set = set()
    
    # Order elements: try larger elements first (they're "safer" to include)
    # since 1/x is small and harder to match as a sum of larger reciprocals
    order = list(range(24, 0, -1))
    
    def backtrack(idx, current_set, current_size):
        nonlocal best_size, best_set
        
        if deadline is not None and time.time() > deadline:
            return
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
        
        # Pruning: even if we include all remaining, can we beat best?
        remaining = len(order) - idx
        if current_size + remaining <= best_size:
            return
        
        if idx == len(order):
            return
        
        x = order[idx]
        
        # Option 1: Include x
        candidate_set = current_set | {x}
        # Forward check: is the new set valid?
        result = _check_valid(candidate_set, deadline)
        if result is True:
            backtrack(idx + 1, candidate_set, current_size + 1)
        elif result is None:
            return
        
        # Option 2: Exclude x
        backtrack(idx + 1, current_set, current_size)
    
    backtrack(0, set(), 0)
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s limit
    
    best_result = set()
    best_size = 0
    
    # Strategy 1: Greedy from high to low
    order1 = list(range(24, 0, -1))
    s1 = _greedy_build(order1, deadline)
    if len(s1) > best_size:
        best_result = s1
        best_size = len(s1)
    
    # Strategy 2: Greedy from low to high
    order2 = list(range(1, 25))
    s2 = _greedy_build(order2, deadline)
    if len(s2) > best_size:
        best_result = s2
        best_size = len(s2)
    
    # Strategy 3: Local search starting from high numbers
    s3 = set(range(13, 25))
    s3_improved = _local_search(s3, deadline)
    if len(s3_improved) > best_size:
        best_result = s3_improved
        best_size = len(s3_improved)
    
    # Strategy 4: Backtracking search (limited by time)
    if time.time() < deadline - 10:
        s4 = _backtracking_max_subset(deadline)
        if len(s4) > best_size:
            best_result = s4
            best_size = len(s4)
    
    # Strategy 5: Random restarts with local search
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        rand_set = set(random.sample(range(1, 25), 15))
        s_improved = _local_search(rand_set, deadline)
        if len(s_improved) > best_size:
            best_result = s_improved
            best_size = len(s_improved)
    
    # Final verification
    if best_result:
        final_check = _check_valid(best_result)
        if final_check is not True:
            # Fallback to a known valid set
            best_result = set(range(20, 25))
    
    return sorted(best_result)

# EVOLVE-BLOCK-END
