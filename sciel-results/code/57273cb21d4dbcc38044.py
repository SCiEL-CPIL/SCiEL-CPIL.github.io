import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    best_set = list(range(13, 25))
    best_size = len(best_set)
    
    def check_validity(S):
        # S is a list of ints
        n = len(S)
        if n == 0:
            return True
        
        # Compute LCM
        l = 1
        for x in S:
            l = l * x // math.gcd(l, x)
            
        # For each x in S, check if L/x is subset sum of L/y for y != x
        for i, x in enumerate(S):
            target = l // x
            # Create list of values for others
            vals = [l // y for y in S if y != x]
            
            # Subset sum check
            # Use a set of possible sums
            # Optimization: if sum(vals) < target, return False for this x? No, if sum < target, target cannot be formed.
            if sum(vals) < target:
                continue
                
            # Meet in the middle or simple set expansion?
            # For n<=24, n-1 <= 23. 2^23 is 8M. Too slow for repeated checks.
            # We need a faster check.
            # However, we only call this on candidate sets.
            # If we are doing greedy, the set size grows.
            # Let's use a simple recursive check with pruning?
            # Or just a set of sums, but break early.
            
            sums = {0}
            for v in vals:
                new_sums = set()
                for s in sums:
                    ns = s + v
                    if ns <= target: # Prune sums > target
                        new_sums.add(ns)
                sums |= new_sums
                if target in sums:
                    return False
            # If we finish loop, target not found for this x
        return True

    # Randomized Greedy
    for _ in range(1000): # Max iterations
        if time.time() > deadline:
            break
            
        perm = list(range(1, 25))
        random.shuffle(perm)
        current_set = []
        
        for x in perm:
            candidate = current_set + [x]
            # Only check if adding x might improve size?
            # We just build the set.
            if check_validity(candidate):
                current_set = candidate
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
        
        # Local search: try to add remaining elements
        remaining = [x for x in range(1, 25) if x not in current_set]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            candidate = current_set + [x]
            if check_validity(candidate):
                current_set = candidate
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                
    return best_set
