import time
from math import gcd
from functools import reduce
import random

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
    
    # Strategy: 
    # 1. Greedy heuristic to find a good initial solution (high best_size) for pruning.
    # 2. Backtracking search with fail-fast collision check.
    
    def check_and_add(sums, v):
        """
        Returns (is_valid, new_sums) if adding value v to set of sums 'sums'
        results in distinct subset sums.
        Fails fast if a collision is detected.
        """
        new_sums = set()
        for s in sums:
            ns = s + v
            # Check collision with existing sums or newly generated sums
            if ns in sums or ns in new_sums:
                return False, None
            new_sums.add(ns)
        return True, sums.union(new_sums)

    def greedy_find(order):
        """
        Greedily add elements in the given order if they maintain dissociation.
        """
        current_sums = {0}
        current_set = []
        for i in order:
            v = values[i]
            is_valid, new_sums = check_and_add(current_sums, v)
            if is_valid:
                current_sums = new_sums
                current_set.append(i)
        return current_set

    # Try a few greedy orders to find a good initial solution
    best_size = 0
    best_set = []
    
    orders = [
        list(reversed(universe)),  # 22, 21, ..., 1
        list(universe),            # 1, 2, ..., 22
        sorted(universe, key=lambda x: values[x]), # Ascending values (22..1)
        sorted(universe, key=lambda x: -values[x]) # Descending values (1..22)
    ]
    
    # Add a few random orders
    random.seed(42)
    for _ in range(5):
        rand_order = universe[:]
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        s = greedy_find(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Backtracking Search
    # Use the order that performed well in greedy, or stick to a standard one.
    # Let's use descending indices (22..1) as it corresponds to ascending values,
    # which might be "safer" for collisions? Or ascending indices?
    # Let's try the order from the best greedy if it was one of the standard ones, 
    # otherwise use reversed universe.
    # Actually, let's just use reversed universe (22..1) for the search as before,
    # but now with a strong initial bound.
    
    candidates = sorted(universe, reverse=True)
    
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
    # We can also start the search from the best_set found by greedy?
    # No, standard backtracking from empty is fine, the pruning will handle it.
    # However, we can inject the best_set into the search state?
    # It's complex to map a set to a path in the tree.
    # Just starting from empty with high best_size is effective.
    
    search(0, {0}, [])
    
    return best_set
