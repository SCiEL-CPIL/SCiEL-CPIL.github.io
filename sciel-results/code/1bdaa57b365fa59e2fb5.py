"""Rewritten program for large reciprocal-subsum-free sets."""
import time
import random
from math import gcd
from functools import reduce

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLEN - (time.time() - _START_TIME) if False else _DEADLINE - (time.time() - _START_TIME)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_subsum_free(S, k):
    """
    Check if 1/k can be expressed as a sum of reciprocals of a non-empty
    subset of S.
    Returns True if it is free (cannot be formed), False otherwise.
    """
    if not S:
        return True
    
    # Calculate LCM of all elements in S and k.
    L = k
    for x in S:
        L = _lcm(L, x)
        # LCM(1..24) is 5354228880, which fits in int.
        
    target = L // k
    
    # Values to sum: L // x for x in S
    vals = [L // x for x in S]
    
    n = len(vals)
    if n == 0:
        return True
        
    # Quick check: if sum(vals) < target, return True (free)
    total_sum = sum(vals)
    if total_sum < target:
        return True
        
    # Filter values > target
    vals = [v for v in vals if v <= target]
    
    if not vals:
        return True
        
    if target in vals:
        return False
        
    # Sort descending for DFS pruning
    vals.sort(reverse=True)
    
    # Precompute suffix sums
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + vals[i]
        
    # DFS
    found = False
    
    def dfs(i, current_sum):
        nonlocal found
        if found:
            return
        if i == n:
            if current_sum == target:
                found = True
            return
            
        # Prune 1: if current_sum > target
        if current_sum > target:
            return
            
        # Prune 2: if current_sum + suffix[i] < target
        if current_sum + suffix[i] < target:
            return
            
        # Option 1: Include vals[i]
        if current_sum + vals[i] == target:
            found = True
            return
        elif current_sum + vals[i] < target:
            dfs(i + 1, current_sum + vals[i])
            if found:
                return
        
        # Option 2: Exclude vals[i]
        dfs(i + 1, current_sum)
        
    dfs(0, 0)
    
    return not found

def _greedy_construct(order):
    """
    Construct a set by greedily adding elements in the given order.
    """
    S = []
    S_set = set()
    for k in order:
        if k in S_set:
            continue
        if _is_subsum_free(S, k):
            S.append(k)
            S_set.add(k)
    return S

def _local_improve(S):
    """
    Try to add any unused element to S.
    Repeat until no more elements can be added.
    """
    S = list(S)
    S_set = set(S)
    improved = True
    while improved:
        improved = False
        for k in range(1, 25):
            if k not in S_set:
                if _is_subsum_free(S, k):
                    S.append(k)
                    S_set.add(k)
                    improved = True
    return S

def construct():
    best_S = []
    
    # 1. Standard Greedy (Descending)
    S_desc = _greedy_construct(range(24, 0, -1))
    S_desc = _local_improve(S_desc)
    if len(S_desc) > len(best_S):
        best_S = S_desc
        
    # 2. Standard Greedy (Ascending)
    S_asc = _greedy_construct(range(1, 25))
    S_asc = _local_improve(S_asc)
    if len(S_asc) > len(best_S):
        best_S = S_asc

    # 3. Randomized Greedy + Local Improve
    # We have time to run many restarts
    num_restarts = 200
    for i in range(num_restarts):
        if _time_remaining() < 5.0:
            break
            
        order = list(range(1, 25))
        random.shuffle(order)
        
        S_rand = _greedy_construct(order)
        S_rand = _local_improve(S_rand)
        
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    if not best_S:
        return list(range(13, 25))
        
    return sorted(best_S)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
