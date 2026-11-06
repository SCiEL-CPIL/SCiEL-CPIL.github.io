import itertools
import math
import time
import random

def construct():
    # Target: Maximize cardinality of dissociated subset of [1, 22]
    N = 22
    candidates = list(range(1, N + 1))
    
    # Calculate LCM of 1..22
    L = 1
    for i in range(1, N + 1):
        L = L * i // math.gcd(L, i)
        
    weights = {i: L // i for i in candidates}
    
    def dfs(pos, current_set, current_sums, K, cands, weights, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return None
            
        if len(current_set) == K:
            return current_set
            
        if pos >= len(cands):
            return None
            
        # Pruning: not enough elements left
        if len(cands) - pos < K - len(current_set):
            return None
            
        for i in range(pos, len(cands)):
            x = cands[i]
            w = weights[x]
            
            # Check if adding x causes collision
            # new_sums = current_sums U (current_sums + w)
            # Collision if len(union) < 2 * len(current_sums)
            
            # Optimization: Instead of creating a new set every time, 
            # we can check intersection.
            # (current_sums + w) intersect current_sums must be empty?
            # No, (current_sums + w) and current_sums can intersect.
            # But (current_sums + w) and (current_sums + w) are distinct?
            # The condition is that all 2^(k+1) sums are distinct.
            # This is equivalent to:
            # 1. current_sums has distinct elements (guaranteed by induction)
            # 2. (current_sums + w) has distinct elements (guaranteed by 1)
            # 3. current_sums and (current_sums + w) are disjoint.
            
            # So we just need to check if (current_sums + w) intersects current_sums.
            # Intersection is empty iff for all s in current_sums, s+w not in current_sums.
            # This is O(2^k).
            
            # Let's compute the shift
            # We can iterate through current_sums and check membership in current_sums.
            # If any s+w is in current_sums, collision.
            
            has_collision = False
            for s in current_sums:
                if (s + w) in current_sums:
                    has_collision = True
                    break
            
            if not has_collision:
                # Construct new sums for recursion
                # We can pass the set, but copying is O(2^k).
                # For k=12, 4096 elements. Copying is fast.
                new_sums = current_sums | {s + w for s in current_sums}
                
                res = dfs(i + 1, current_set + [x], new_sums, K, cands, weights, start_time, time_limit)
                if res is not None:
                    return res
                    
        return None

    start_time = time.time()
    time_limit = 50.0 # Leave 10s buffer
    
    # Try K from 12 down to 1
    # We can randomize the candidate order to escape local minima/bad branches
    best_set = [1] # Fallback
    
    # First, try a deterministic order (descending value)
    # Large numbers -> small weights -> less likely to cause small sum collisions?
    # Actually, let's try a few orders.
    
    orders = [
        list(reversed(candidates)), # 22..1
        candidates, # 1..22
        # Maybe a mixed order?
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]
    
    found = None
    for K in range(12, 0, -1):
        for order in orders:
            # Shuffle slightly? No, keep deterministic first.
            res = dfs(0, [], {0}, K, order, weights, start_time, time_limit)
            if res is not None:
                found = res
                break
        if found:
            break
        # If time is running out, stop trying smaller K? 
        # No, we want max K. If we fail K=12, we must try K=11.
        
    if found:
        return found
    else:
        # If DFS fails (unlikely for small K), fallback to greedy or seed
        # The seed [1, 2, 3, 5, 7, 11, 13, 17, 19] is size 9.
        # Let's verify it's valid and return it if no better found.
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
