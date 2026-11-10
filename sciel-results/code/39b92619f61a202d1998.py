import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Calculate LCM of 1..24
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 25))
    
    # Precompute integer values for 1/x * L
    vals = [0] + [L // i for i in range(1, 25)]
    
    # Helper: Check if target can be formed by a subset of items (integers)
    def can_form(target, items):
        if not items:
            return target == 0
        
        n = len(items)
        mid = n // 2
        left_items = items[:mid]
        right_items = items[mid:]
        
        # Generate all subset sums for left
        left_sums = {0}
        for item in left_items:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + item)
            left_sums.update(new_sums)
            
        # Generate all subset sums for right
        right_sums = {0}
        for item in right_items:
            new_sums = set()
            for s in right_sums:
                new_sums.add(s + item)
            right_sums.update(new_sums)
        
        # Check if any pair sums to target
        if len(left_sums) < len(right_sums):
            for s in left_sums:
                if target - s in right_sums:
                    return True
        else:
            for s in right_sums:
                if target - s in left_sums:
                    return True
            
        return False

    def is_valid(S):
        """
        Check if set S is valid.
        S is a list of integers 1..24.
        """
        for x in S:
            target = vals[x]
            others = [vals[y] for y in S if y != x]
            if can_form(target, others):
                return False
        return True

    best_set = []
    best_size = 0
    
    # Initial greedy: Large to Small
    current_set = []
    for x in range(24, 0, -1):
        if is_valid(current_set + [x]):
            current_set.append(x)
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(current_set)
        
    # Initial greedy: Small to Large
    current_set = []
    for x in range(1, 25):
        if is_valid(current_set + [x]):
            current_set.append(x)
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(current_set)

    # Randomized Greedy Search
    elements = list(range(1, 25))
    
    while time.time() < deadline:
        random.shuffle(elements)
        current_set = []
        for x in elements:
            if is_valid(current_set + [x]):
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            if best_size >= 22:
                break

    # Local Search on best_set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_best = best_set[:]
        
        # Try adding any single element
        for x in range(1, 25):
            if x not in current_best:
                if is_valid(current_best + [x]):
                    current_best.append(x)
                    improved = True
                    best_set = current_best
                    best_size = len(current_best)
                    break
        
        if not improved:
             # Try 1-for-1 swaps to potentially unblock
             for r in current_best:
                 for u in range(1, 25):
                     if u not in current_best:
                         candidate = [x for x in current_best if x != r] + [u]
                         if is_valid(candidate):
                             # Check if this swap allows adding something else
                             added_any = False
                             for v in range(1, 25):
                                 if v not in candidate:
                                     if is_valid(candidate + [v]):
                                         candidate.append(v)
                                         best_set = candidate
                                         best_size = len(candidate)
                                         improved = True
                                         added_any = True
                                         break
                             if added_any:
                                 break
                 if improved:
                     break
                     
    if not best_set:
        return [24]
        
    return best_set

if __name__ == "__main__":
    result = construct()
    print(result)
