"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    
    # Precompute 3-APs
    # For each x, we need to know which pairs (y, z) form an AP with x.
    # Specifically, if we are considering adding x to a set S, 
    # we must ensure that for no (y, z) in conflict_pairs[x] are both y and z in S.
    
    # To make B&B efficient, we process elements in order of decreasing weight (1/x).
    # So we consider x = 1, 2, ..., n.
    # Decision at step i: Include x_i or not?
    
    # Precompute conflict pairs for fast checking.
    # conflict_pairs[x] = list of (y, z) such that {x, y, z} is a 3-AP and y, z < x?
    # No, y and z can be anything.
    # But in B&B, we decide on x in increasing order.
    # If we decide to INCLUDE x, we must check against already INCLUDED elements.
    # If we decide to EXCLUDE x, no check needed.
    
    # Let's refine the check for "Can x be added to current_set?"
    # current_set contains elements < x (since we process in order).
    # We need to check if there exists y, z in current_set such that {x, y, z} is a 3-AP.
    # This happens if:
    # 1. x is the middle term: y < x < z. But z > x, so z is not in current_set. 
    #    Wait, if we process 1..n, and current_set has elements < x, then z cannot be in current_set if z > x.
    #    So x can never be the middle term of an AP formed with elements strictly smaller than x?
    #    Let's check: y < x < z. If y, z in S, and S subset {1..x-1}, then z < x. Contradiction.
    #    So if we build the set by adding elements in increasing order, 
    #    a new element x can only form an AP with previous elements if x is the LARGEST element of the AP.
    #    i.e., y < z < x and y + x = 2z => x = 2z - y.
    #    Or x is the SMALLEST? No, x is the new largest.
    #    So we only need to check if there exist y, z in current_set such that x = 2z - y.
    #    This is equivalent to: for each z in current_set, check if y = 2z - x is in current_set and y < z.
    
    # This simplifies the check significantly!
    # Check(x, current_set):
    #   for z in current_set:
    #       y = 2*z - x
    #       if y < z and y in current_set:
    #           return False
    #   return True
    
    # This check is O(|current_set|).
    
    # B&B Structure:
    # We want to maximize Sum(1/x).
    # We traverse a binary tree of decisions: Include 1, Exclude 1, ... Include 2, Exclude 2...
    # State: (current_index, current_sum, current_set)
    # Upper Bound: current_sum + Sum(1/x for x in [current_index+1, n])
    # If Upper Bound <= best_sum_found, prune.
    
    # To make this efficient in Python:
    # 1. Use a boolean array for current_set for O(1) membership.
    # 2. Keep a list of included elements to iterate in the check.
    # 3. Precompute suffix sums of reciprocals for the upper bound.
    
    # Precomputation
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i
        
    suffix_sums = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[i]
        
    best_sum = 0.0
    best_set = []
    
    # To avoid deep recursion issues or overhead, we can use a stack or just recursive with optimization.
    # Given n=256, depth is 256. Python recursion limit is usually 1000, so it's fine.
    # But function call overhead might be high. 
    # However, pruning should be very effective.
    
    # Let's use a recursive function.
    # To speed up, we can also use a heuristic to find a good initial solution (Greedy).
    
    # Greedy Initialization
    greedy_set = []
    greedy_bool = [False] * (n + 1)
    current_sum_greedy = 0.0
    for x in range(1, n + 1):
        can_add = True
        # Check if x forms AP with any pair in greedy_set
        # Since x is the largest, check y = 2z - x for z in greedy_set
        for z in greedy_set:
            y = 2 * z - x
            if y < z and greedy_bool[y]:
                can_add = False
                break
        if can_add:
            greedy_set.append(x)
            greedy_bool[x] = True
            current_sum_greedy += weights[x]
            
    best_sum = current_sum_greedy
    best_set = greedy_set[:]
    
    # B&B Search
    # We will search for a solution better than best_sum.
    # We process elements 1..n.
    
    # Optimization: 
    # If we are at index i, and the remaining potential sum (suffix_sums[i+1]) plus current_sum 
    # is <= best_sum, we stop.
    
    # We also need to track the current set for the AP check.
    # Using a list and a boolean array.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def solve(idx, current_sum, current_set_list, current_bool_array):
        nonlocal best_sum, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: Upper bound check
        # Max possible sum from idx to n
        # Note: idx is the NEXT element to consider.
        # So remaining elements are idx..n.
        # Wait, in my loop structure, I usually decide on idx, then move to idx+1.
        # Let's define: 
        # idx: current element index (1-based).
        # If idx > n, we have a complete solution.
        
        if idx > n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set_list[:]
            return
            
        # Upper Bound: current_sum + suffix_sums[idx]
        # If this is <= best_sum, no point continuing.
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Option 1: Include idx
        # Check if idx can be added to current_set_list
        can_add = True
        for z in current_set_list:
            y = 2 * z - idx
            if y < z and current_bool_array[y]:
                can_add = False
                break
                
        if can_add:
            current_set_list.append(idx)
            current_bool_array[idx] = True
            solve(idx + 1, current_sum + weights[idx], current_set_list, current_bool_array)
            current_set_list.pop()
            current_bool_array[idx] = False
            
        # Option 2: Exclude idx
        # Even if we exclude, we might still beat best_sum with subsequent elements.
        solve(idx + 1, current_sum, current_set_list, current_bool_array)
        
    # Start B&B
    # Initial state: idx=1, sum=0, set=[], bool=[False]*(n+1)
    # Note: We already have a good best_sum from greedy.
    
    # One optimization: 
    # If 1 is not in the optimal set, the max sum is bounded by sum(1/2..1/n).
    # The greedy set usually contains 1.
    # If greedy contains 1, it's likely optimal contains 1.
    # But B&B will verify.
    
    # To make it faster, we can branch on "Include" first, as smaller numbers have higher weight.
    
    initial_set = []
    initial_bool = [False] * (n + 1)
    
    solve(1, 0.0, initial_set, initial_bool)
    
    # Return the best set found
    # Ensure it's a list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
