"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1), 1)

# Precompute LCM of 1..24
LCM_24 = _lcm_range(24)

def _is_free(s):
    """Check if set s is reciprocal-subsum-free using exact integer arithmetic."""
    s_set = set(s)
    n = len(s_set)
    if n <= 1:
        return True
    
    # For each x in s, check if LCM/x equals sum of LCM/y for any non-empty subset of s\{x}
    for x in s_set:
        target = LCM_24 // x
        others = [y for y in s_set if y != x]
        if not others:
            continue
        
        # We need to check if target can be written as sum of {LCM_24 // y : y in others}
        # This is a subset sum problem
        values = [LCM_24 // y for y in others]
        m = len(values)
        
        # Sort values in descending order for better pruning
        values_sorted = sorted(values, reverse=True)
        
        # Suffix sum for pruning
        suffix = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] + values_sorted[i]
        
        # DFS with memoization on (idx, remaining) - but remaining can be large
        # Use simple DFS with pruning
        found = [False]
        
        def dfs(idx, curr):
            if found[0]:
                return
            if curr == target:
                if idx > 0:
                    found[0] = True
                return
            if idx == m:
                return
            if curr > target:
                return
            if curr + suffix[idx] < target:
                return
            
            # Include values_sorted[idx]
            new_curr = curr + values_sorted[idx]
            if new_curr <= target:
                dfs(idx + 1, new_curr)
                if found[0]:
                    return
            # Exclude
            dfs(idx + 1, curr)
        
        dfs(0, 0)
        if found[0]:
            return False
    return True

def _greedy_build(order):
    """Build a set by greedily adding elements in the given order."""
    s = set()
    for x in order:
        candidate = s | {x}
        if _is_free(candidate):
            s = candidate
    return s

def _local_search(s, time_limit):
    """Try to improve set s by adding or removing elements."""
    s = set(s)
    best = s
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try adding each element not in s
        for x in range(1, 25):
            if x not in s:
                candidate = s | {x}
                if _is_free(candidate):
                    s = candidate
                    best = s
                    improved = True
                    break
        
        if not improved:
            # Try removing one element and adding two
            # Or just try swapping
            for x in list(s):
                if time.time() - start > time_limit:
                    break
                s_without = s - {x}
                for y in range(1, 25):
                    if y not in s:
                        candidate = s_without | {y}
                        if _is_free(candidate):
                            s = candidate
                            best = s
                            improved = True
                            break
                    if time.time() - start > time_limit:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    best_set = set(range(13, 25))  # Seed fallback
    
    # Try multiple greedy orders
    orders = [
        list(range(24, 0, -1)),  # High to low
        list(range(1, 25)),      # Low to high
        list(range(24, 0, -1, 2)),  # Even high to low
        list(range(23, 0, -2)),  # Odd high to low
    ]
    
    # Add some random orders
    random.seed(42)
    for _ in range(10):
        order = list(range(1, 25))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() - start_time > timeout:
            break
        s = _greedy_build(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search on best set found
    if time.time() - start_time < timeout:
        remaining = timeout - (time.time() - start_time)
        improved = _local_search(best_set, remaining)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Final verification
    if _is_free(best_set):
        return sorted(best_set)
    
    # Fallback
    return list(range(13, 25))

# EVOLVE-BLOCK-END
