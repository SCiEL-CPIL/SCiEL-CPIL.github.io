import time
from math import lcm
from functools import reduce
import random

def construct():
    start_time = time.time()
    # Total time limit is 60s. We use 55s for search.
    search_deadline = start_time + 55.0
    
    L = reduce(lcm, range(1, 23))
    V = [L // i for i in range(1, 23)]
    
    best_set = None
    
    def search_k(K, order, time_limit):
        current_sums = {0}
        current_set = []
        deadline = time.time() + time_limit
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if len(current_set) == K:
                return current_set[:]
            
            # Pruning: not enough elements left
            if 22 - idx < K - len(current_set):
                return None
            
            # Option 1: Include order[idx]
            v = V[order[idx]]
            # Check for collision
            # new_sums = {s + v for s in current_sums}
            # if current_sums & new_sums: collision
            new_sums = {s + v for s in current_sums}
            if not (current_sums & new_sums):
                old_sums = current_sums
                current_sums = old_sums | new_sums
                current_set.append(order[idx])
                
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                # Backtrack
                current_sums = old_sums
                current_set.pop()
            
            # Option 2: Exclude order[idx]
            res = backtrack(idx + 1)
            if res is not None:
                return res
            
            return None
        
        return backtrack(0)

    random.seed(42)
    
    # Try K from 22 down to 1?
    # We know 16 is possible.
    # Let's try 18, 17, 16.
    # If we find 18, great. If not, 17. If not, 16.
    
    # We can also try 22, 21, 20 with very short times to see if lucky.
    # But 18 is more likely to be the max.
    
    targets = [(18, 10.0), (17, 15.0), (16, 25.0)]
    
    for K, t_limit in targets:
        if time.time() > search_deadline - 5.0:
            break
            
        found = None
        # Try natural order first
        order = list(range(22))
        found = search_k(K, order, t_limit)
        
        if found:
            best_set = found
            break
        
        # Try random orders
        remaining_time = min(t_limit, search_deadline - time.time() - 2.0)
        if remaining_time > 2.0:
            attempts = 0
            while time.time() < search_deadline - 2.0 and attempts < 10:
                order = list(range(22))
                random.shuffle(order)
                # Give each attempt a small time slice
                # Dynamic slice
                slice_time = max(0.5, (search_deadline - time.time() - 2.0) / (10 - attempts))
                found = search_k(K, order, slice_time)
                if found:
                    best_set = found
                    break
                attempts += 1
            if found:
                break

    if best_set is None:
        # Fallback to a known valid set or [1]
        # The seed set [1, 2, 3, 5, 7, 11, 13, 17, 19] is valid?
        # Let's just use [1] to be safe, or try to construct a small one.
        # Actually, we can just return [1].
        best_set = [0] # Index 0 corresponds to 1
        
    # Convert indices to values 1..22
    result = [i + 1 for i in best_set]
    
    # Final validation
    def is_valid(subset):
        if not subset:
            return False
        vals = [L // x for x in subset]
        sums = {0}
        for v in vals:
            new_sums = {s + v for s in sums}
            if sums & new_sums:
                return False
            sums |= new_sums
        return len(sums) == 2 ** len(subset)
        
    if not is_valid(result):
        result = [1]
        
    return result

