import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Helper to check if a set is valid (no subset sums to 200)
    def get_reachable_sums(S):
        reachable = {0}
        for x in S:
            new_sums = {s + x for s in reachable if s + x <= 200}
            if 200 in new_sums:
                return None # Invalid
            reachable |= new_sums
        return reachable
    
    # Initial solution: {51, ..., 100}
    current_set = set(range(51, 101))
    current_size = len(current_set)
    
    # Verify initial set
    init_reachable = get_reachable_sums(current_set)
    if init_reachable is None:
        # Fallback to greedy
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected
    
    best_set = set(current_set)
    best_size = current_size
    
    # Strategy: Try to find a valid set of size 51 by removing one element from [51,100]
    # and adding two elements from [1,50].
    
    candidates_small = list(range(1, 51))
    candidates_large = list(range(51, 101))
    
    # Precompute: for each large element L, which pairs (x,y) from small would make the set invalid?
    # This is expensive, so we'll do it on the fly but with early termination.
    
    # Better approach: For each L in [51,100], try all pairs (x,y) from [1,50]
    # and check if current_set - {L} + {x,y} is valid.
    
    # To speed up, note that if we remove L, the reachable sums of current_set - {L} 
    # are a subset of reachable sums of current_set.
    
    # Let's precompute reachable sums for current_set
    base_reachable = get_reachable_sums(current_set)
    
    # For each L, compute reachable sums without L
    # This is tricky because reachable sums are not easily "uncomputed"
    
    # Alternative: For each L, recompute from scratch (fast enough for 50 elements)
    # But 50 * C(50,2) = 50 * 1225 = 61250 checks, each O(51*200) = ~10200 ops
    # Total: ~625M ops, might be too slow in Python.
    
    # Optimized: For each L, first check if removing L and adding x,y could possibly work
    # by checking if 200 - x - y is reachable in current_set - {L}
    
    # Let's try a randomized approach with more iterations and better heuristics
    
    max_iters = 10000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Randomly pick a large element to remove
        L = random.choice(candidates_large)
        
        # Randomly pick two small elements to add
        x, y = random.sample(candidates_small, 2)
        
        # Proposed new set
        new_set = set(current_set)
        new_set.remove(L)
        new_set.add(x)
        new_set.add(y)
        
        # Check validity with early termination
        reachable = {0}
        valid = True
        for z in sorted(new_set, reverse=True):  # Process larger elements first for early termination
            new_sums = {s + z for s in reachable if s + z <= 200}
            if 200 in new_sums:
                valid = False
                break
            reachable |= new_sums
            
        if valid:
            return list(new_set) # Found size 51!
    
    # If no improvement found, return best set (size 50)
    return list(best_set)
