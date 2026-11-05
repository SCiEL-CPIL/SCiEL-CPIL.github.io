import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Universe 1..22
    universe = list(range(1, 23))
    
    # Calculate LCM of 1..22 to scale reciprocals to integers
    # LCM(1..22) = 232792560
    L = reduce(_lcm, universe, 1)
    
    # Map each number to its scaled reciprocal
    # v_i = L / i
    values = {i: L // i for i in universe}
    
    # We want to find the largest subset S such that all subset sums of {v_i | i in S} are distinct.
    # This is equivalent to: for any distinct subsets A, B of S, sum(A) != sum(B).
    # Which is equivalent to: the set of all subset sums has size 2^{|S|}.
    
    # Strategy: Backtracking search.
    # We order the universe to improve pruning.
    # Let's try descending order: 22, 21, ..., 1.
    # Smaller reciprocals (larger i) might be "safer" to add?
    # Or smaller reciprocals accumulate to smaller sums, potentially avoiding large collisions?
    # Let's stick with descending.
    candidates = sorted(universe, reverse=True)
    
    best_size = 0
    best_set = []
    
    # To allow for better pruning, we can also try different orderings if the first one is slow,
    # but with N=22, the search space is small enough for a single good ordering to likely succeed fast.
    
    def check_and_add(sums, v):
        """
        Returns (is_valid, new_sums) if adding value v to set of sums 'sums'
        results in distinct subset sums.
        """
        new_sums = set()
        for s in sums:
            ns = s + v
            new_sums.add(ns)
        
        # Check for collision with existing sums
        if sums.isdisjoint(new_sums):
            return True, sums.union(new_sums)
        else:
            return False, None

    def search(idx, current_sums, current_set):
        nonlocal best_size, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # Pruning: if we can't beat best size even by taking all remaining
        remaining = len(candidates) - idx
        if len(current_set) + remaining <= best_size:
            return
            
        if idx >= len(candidates):
            return
            
        v = values[candidates[idx]]
        
        # Branch 1: Include candidates[idx]
        # Check if adding v keeps subset sums distinct
        is_valid, new_sums = check_and_add(current_sums, v)
        if is_valid:
            search(idx + 1, new_sums, current_set + [candidates[idx]])
            
        # Branch 2: Exclude candidates[idx]
        search(idx + 1, current_sums, current_set)
        
    # Initial state: empty set, sums = {0}
    search(0, {0}, [])
    
    # If the search didn't finish or found nothing (shouldn't happen), 
    # we might want to try a different ordering or greedy fallback, 
    # but N=22 is small enough that the above should find the optimal or near-optimal quickly.
    
    # Fallback: If best_set is empty (should not happen as singletons are always valid), 
    # or if we want to be safe, we can return a known good set.
    # But the search above is exhaustive with pruning.
    
    return best_set
