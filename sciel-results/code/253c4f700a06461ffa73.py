import time
import random
from math import gcd
from functools import reduce

def construct():
        start_time = time.time()
        deadline = start_time + 55.0
        
        # Precompute LCM and targets
        # LCM(1..24)
        lcm_all = 53542288800
        targets = {i: lcm_all // i for i in range(1, 25)}
        
        def can_make_sum(target, vals):
            # MITM
            n = len(vals)
            if n == 0:
                return target == 0
            if target == 0:
                return True
            if sum(vals) < target:
                return False
            
            mid = n // 2
            left = vals[:mid]
            right = vals[mid:]
            
            left_sums = set([0])
            for v in left:
                new_sums = set()
                for s in left_sums:
                    new_sums.add(s + v)
                left_sums.update(new_sums)
                
            right_sums = set([0])
            for v in right:
                new_sums = set()
                for s in right_sums:
                    new_sums.add(s + v)
                right_sums.update(new_sums)
                
            for s in left_sums:
                if (target - s) in right_sums:
                    return True
            return False

        def is_valid(S):
            S_list = list(S)
            for x in S_list:
                target = targets[x]
                others = [targets[y] for y in S_list if y != x]
                if can_make_sum(target, others):
                    return False
            return True

        best_set = set(range(13, 25))
        best_size = len(best_set)
        
        # Greedy add
        current_set = set(best_set)
        candidates = list(range(1, 13))
        # Sort candidates: try larger numbers first (smaller reciprocals, less likely to be sum of others? 
        # Actually, smaller numbers (larger reciprocals) are more likely to be formed by sums of small reciprocals?
        # No, large reciprocals are hard to form from small ones.
        # Small reciprocals (large numbers) are easy to form? No.
        # Let's just try all.
        
        while True:
            added = False
            for x in candidates:
                if x in current_set:
                    continue
                if time.time() > deadline:
                    return list(best_set)
                new_set = current_set | {x}
                if is_valid(new_set):
                    current_set = new_set
                    added = True
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(best_set)
                    break # Restart candidate loop to try next
            if not added:
                break

        # Local Search: Swaps
        found_improvement = True
        while found_improvement and time.time() < deadline:
            found_improvement = False
            # Try to replace an element to allow adding another?
            # Simple swap: remove x, add y.
            # This keeps size same. We need to increase size.
            # So: Remove x, Add y, Add z?
            
            # Strategy: Try to remove one element, then greedily add as many as possible.
            for x in list(current_set):
                if time.time() > deadline:
                    break
                reduced_set = current_set - {x}
                # Greedy add to reduced_set
                temp_set = set(reduced_set)
                for y in range(1, 25):
                    if y in temp_set:
                        continue
                    new_set = temp_set | {y}
                    if is_valid(new_set):
                        temp_set = new_set
                
                if len(temp_set) > best_size:
                    best_set = set(temp_set)
                    best_size = len(best_set)
                    current_set = set(temp_set)
                    found_improvement = True
                    break

        # Random Restarts
        for _ in range(100):
            if time.time() > deadline:
                break
            # Start with a random subset of size 10-12?
            # Or start with best_set and perturb
            current_set = set(best_set)
            # Remove a random element
            if current_set:
                x_to_remove = random.choice(list(current_set))
                current_set.remove(x_to_remove)
            
            # Greedy add
            for x in range(1, 25):
                if x in current_set:
                    continue
                new_set = current_set | {x}
                if is_valid(new_set):
                    current_set = new_set
            
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)

        return list(best_set)
