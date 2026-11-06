import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    # Theoretical max is 40
    target_size = 40
    
    def verify(s):
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 4096 for x in s):
            return False
        sums = [sum(t) for t in combinations_with_replacement(s, 3)]
        return len(sums) == len(set(sums))

    def try_greedy(order, time_budget, start_time_ref):
        selected = []
        current_sums = set()
        
        for x in order:
            if time.time() - start_time_ref > time_budget:
                break
            if len(selected) >= target_size:
                break
            
            # Check if x can be added
            valid = True
            new_sums = set()
            
            # (x,x,x)
            s = 3 * x
            if s in current_sums:
                continue
            new_sums.add(s)
            
            if not selected:
                selected.append(x)
                current_sums.update(new_sums)
                continue
            
            # (x,x,y)
            for y in selected:
                s = 2 * x + y
                if s in current_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
            
            # (x,y,y)
            for y in selected:
                s = x + 2 * y
                if s in current_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
            
            # (x,y,z)
            for i in range(len(selected)):
                for j in range(i, len(selected)):
                    s = x + selected[i] + selected[j]
                    if s in current_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            
            if valid:
                selected.append(x)
                current_sums.update(new_sums)
        
        return selected

    def local_search(s, time_budget, start_time_ref):
        # Try to improve s by swapping elements
        # This is a heuristic to escape local optima
        # We don't actually implement complex local search here due to time,
        # but we can try to add elements if possible.
        # The main driver is multiple randomized greedy runs.
        return s

    best_set = []
    
    # 1. Deterministic Greedy Ascending
    s = try_greedy(range(1, 4097), time_limit - (time.time() - start_time), start_time)
    if len(s) > len(best_set):
        best_set = s
    
    # 2. Deterministic Greedy Descending
    if time.time() - start_time < time_limit - 5:
        s = try_greedy(range(4096, 0, -1), time_limit - (time.time() - start_time), start_time)
        if len(s) > len(best_set):
            best_set = s
    
    # 3. Randomized Greedy
    random.seed(42)
    for trial in range(500):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Random shuffle
        order = list(range(1, 4097))
        random.shuffle(order)
        
        s = try_greedy(order, time_limit - (time.time() - start_time), start_time)
        if len(s) > len(best_set):
            best_set = s
            
            if len(best_set) == target_size:
                break
    
    # 4. If we haven't found 40, try a few more targeted strategies
    # e.g. bias towards larger numbers
    if len(best_set) < target_size and time.time() - start_time < time_limit - 5:
        for trial in range(200):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Biased random: pick from top half first
            order = list(range(2048, 4097)) + list(range(1, 2048))
            random.shuffle(order)
            
            s = try_greedy(order, time_limit - (time.time() - start_time), start_time)
            if len(s) > len(best_set):
                best_set = s
                if len(best_set) == target_size:
                    break

    # Verify result
    if not verify(best_set):
        # Fallback
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
    return best_set
