import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    candidates = list(range(1, N + 1))
    
    # Calculate LCM and scaled values
    L = 1
    for x in candidates:
        L = math.lcm(L, x)
        
    vals = {x: L // x for x in candidates}
    
    # Greedy to find a lower bound
    greedy_set = []
    greedy_sums = {0}
    for x in candidates:
        v = vals[x]
        new_sums = {s + v for s in greedy_sums}
        if greedy_sums.isdisjoint(new_sums):
            greedy_set.append(x)
            greedy_sums = greedy_sums | new_sums
        # else: skip x
            
    best_size = len(greedy_set)
    best_set = greedy_set
    
    # Try to find larger sets using backtracking
    # Start from N down to best_size + 1
    # To optimize, we can sort candidates by value descending (i.e., index ascending)
    # This is already the case for 1..22.
    
    def backtrack(current_idx, current_set, current_sums, target_size):
        if time.time() - start_time > timeout:
            return None
            
        if len(current_set) == target_size:
            return current_set
            
        if current_idx > N:
            return None
            
        if len(current_set) + (N - current_idx + 1) < target_size:
            return None
            
        # Try to add candidates from current_idx to N
        for i in range(current_idx, N + 1):
            x = i
            v = vals[x]
            
            # Check if adding x creates a collision
            # new_sums = current_sums | {s + v for s in current_sums}
            # Collision if current_sums and {s + v for s in current_sums} intersect
            
            # Optimization: Instead of creating the full set, check intersection
            # We can iterate through current_sums and check if s+v is in current_sums
            # But creating the set is O(2^k). Intersection is O(2^k).
            # For k up to 16, 2^16 = 65536. This is fast.
            
            new_sums = {s + v for s in current_sums}
            if current_sums.isdisjoint(new_sums):
                result = backtrack(i + 1, current_set + [x], current_sums | new_sums, target_size)
                if result is not None:
                    return result
            # else: skip x
            
        return None

    # Try to find a set of size best_size + 1, then + 2, etc.
    # Or start from N down?
    # Starting from N is risky if it's infeasible.
    # Starting from best_size + 1 is safer and likely to succeed if a larger set exists.
    
    for k in range(best_size + 1, N + 1):
        if time.time() - start_time > timeout - 5:
            break
            
        result = backtrack(1, [], {0}, k)
        if result is not None:
            best_set = result
            best_size = k
            # We found a larger set. We can continue to try for even larger.
            # But since we go up, we should stop when we can't find k+1?
            # Or just continue.
            
    return best_set
