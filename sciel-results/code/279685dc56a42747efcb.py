"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce
from fractions import Fraction

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_all(nums):
    if not nums:
        return 1
    return reduce(_lcm, nums)

def _is_valid_fast(S, deadline=None):
    """
    Check if S is reciprocal-subsum-free using LCM-based integer arithmetic.
    Returns True if valid, False if not, None if timeout.
    """
    if not S:
        return True
    
    S_list = list(S)
    n = len(S_list)
    if n <= 1:
        return True
    
    L = _lcm_all(S_list)
    
    for x in S_list:
        if deadline is not None and time.time() > deadline:
            return None
        target = L // x
        others = [y for y in S_list if y != x]
        if not others:
            continue
        vals = [L // y for y in others]
        
        # Sum of all vals
        total_sum = sum(vals)
        if total_sum < target:
            continue
        
        # Use a set of achievable sums, pruning sums > target
        achievable = {0}
        for v in vals:
            if v > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return False
    return True

def _is_valid_exact(S, deadline=None):
    """
    Exact check using Fraction for correctness verification.
    """
    if not S:
        return True
    
    S_list = list(S)
    for x in S_list:
        if deadline is not None and time.time() > deadline:
            return None
        target = Fraction(1, x)
        others = [y for y in S_list if y != x]
        if not others:
            continue
        m = len(others)
        others_sorted = sorted(others)
        
        # Suffix sum for pruning
        suffix_sum = [Fraction(0, 1)] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, others_sorted[i])
        
        if suffix_sum[0] < target:
            continue
        
        found = [False]
        
        def dfs(idx, curr):
            if found[0]:
                return
            if deadline is not None and time.time() > deadline:
                return
            if curr == target:
                if idx > 0:
                    found[0] = True
                return
            if idx == m:
                return
            if curr > target:
                return
            if curr + suffix_sum[idx] < target:
                return
            
            y = others_sorted[idx]
            new_curr = curr + Fraction(1, y)
            if new_curr <= target:
                dfs(idx + 1, new_curr)
                if found[0]:
                    return
            dfs(idx + 1, curr)
        
        dfs(0, Fraction(0, 1))
        if found[0]:
            return False
    return True

def _greedy_decreasing():
    """
    Greedy construction from 24 down to 1.
    Includes x if 1/x is not a subset sum of larger selected elements.
    """
    # Calculate LCM of 1..24
    L = _lcm_all(range(1, 25))
    # Values for each element
    vals = {i: L // i for i in range(1, 25)}
    
    selected = set()
    current_sums = {0}
    
    for x in range(24, 0, -1):
        v = vals[x]
        # Check if v is in current_sums
        if v not in current_sums:
            selected.add(x)
            # Update sums
            new_sums = set(current_sums)
            for s in current_sums:
                ns = s + v
                if ns <= L: # Prune sums > L (though L is max possible sum of all)
                    new_sums.add(ns)
            current_sums = new_sums
        # else: cannot include x, skip
    
    return selected

def _branch_and_bound(deadline):
    """
    Branch and bound to find maximum cardinality set.
    """
    L = _lcm_all(range(1, 25))
    vals = [L // i for i in range(24, 0, -1)] # Index 0 is 24, index 23 is 1
    elems = list(range(24, 0, -1))
    
    best_size = 0
    best_set = set()
    
    # Initial greedy to set lower bound
    greedy_set = _greedy_decreasing()
    best_size = len(greedy_set)
    best_set = greedy_set
    
    # Pre-compute suffix counts for upper bound
    # At index i, we can add at most (24 - i) more elements
    # So upper bound is current_size + (24 - i)
    
    def dfs(idx, current_sums, current_set, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        # Upper bound pruning
        if current_size + (24 - idx) <= best_size:
            return
            
        if idx == 24:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        v = vals[idx]
        elem = elems[idx]
        
        # Check if we can include elem
        can_include = (v not in current_sums)
        
        if can_include:
            # Branch 1: Include elem
            new_sums = set(current_sums)
            for s in current_sums:
                ns = s + v
                if ns <= L:
                    new_sums.add(ns)
            
            current_set.add(elem)
            dfs(idx + 1, new_sums, current_set, current_size + 1)
            current_set.remove(elem)
            
        # Branch 2: Exclude elem
        dfs(idx + 1, current_sums, current_set, current_size)
        
    dfs(0, {0}, set(), 0)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Try Branch and Bound
    # If B&B finishes or times out, it returns the best found
    best_set = _branch_and_bound(deadline)
    
    # Verify the result
    if best_set:
        verify = _is_valid_fast(best_set, deadline)
        if verify is True:
            return sorted(best_set)
        elif verify is None:
            # Timeout on verification, return as-is
            return sorted(best_set)
        else:
            # Invalid, fallback to greedy
            greedy_set = _greedy_decreasing()
            if greedy_set:
                return sorted(greedy_set)
    
    # Fallback
    return list(range(13, 25))

# EVOLVE-BLOCK-END
