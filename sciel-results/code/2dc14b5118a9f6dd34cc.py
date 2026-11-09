import time
import random
import math
from itertools import combinations

_LCM = 232792560
_WEIGHTS = [_LCM // i for i in range(1, 23)]

def _is_dissociated(elements):
    """Check if all subset sums of integer weights are distinct."""
    n = len(elements)
    if n == 0:
        return True
    ws = [_WEIGHTS[e - 1] for e in elements]
    # Sort weights to potentially help with pruning? 
    # Actually, order doesn't matter for the set of sums.
    sums = {0}
    for w in ws:
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        if new_sums & sums:
            return False
        sums |= new_sums
    return True

def _find_max_dissociated(deadline):
    all_elements = list(range(1, 23))
    best_set = []
    
    # Try to find sets of size 17, 18, ...
    # Start with high sizes
    for target_size in range(22, 1, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Randomized search for this size
        # Use local search starting from random subsets of size target_size
        
        # If target_size is small enough, we can do exhaustive
        # But 17 is too big for exhaustive in Python within 90s likely.
        # We use randomized local search.
        
        # Strategy: 
        # 1. Generate random subset of size target_size.
        # 2. Check if dissociated.
        # 3. If not, try to repair by swapping.
        
        # Better: Start with a known good set of size 16 and try to expand.
        # But we don't know a good 16 set initially.
        
        # Let's do a greedy construction first to get a baseline.
        # Sort by weight descending.
        indices_by_weight = sorted(range(1, 23), key=lambda x: _WEIGHTS[x-1], reverse=True)
        
        # Greedy
        current = []
        for idx in indices_by_weight:
            if time.time() > deadline - 5:
                break
            if _is_dissociated(current + [idx]):
                current.append(idx)
        
        if len(current) > len(best_set):
            best_set = current
            
        # Now, try to improve using local search
        # We will run a simulated annealing or hill climbing
        # Objective: Maximize size.
        
        # If we found a set of size target_size, we can return it immediately?
        # No, we want the max.
        
        # Let's try to find a set of size target_size using random restarts.
        # For target_size = 17, 18, etc.
        
        if target_size <= 17:
            # Try 1000 random restarts
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                random.shuffle(all_elements)
                current = all_elements[:target_size]
                if _is_dissociated(current):
                    best_set = current
                    found = True
                    break
                
                # If not dissociated, try to repair?
                # Or just restart.
                
        # If we found a set of this size, we can stop searching for smaller sizes?
        # No, we want the MAX size. So if we found 17, we should check 18 first?
        # The loop goes 22 down to 2.
        # So if we find 17, we continue to check 16? No, we break?
        # No, if we find 17, that's better than 16.
        # But we should check if 18 is possible.
        # The loop starts at 22.
        # So if 22 fails, 21 fails... 18 fails, 17 succeeds -> return 17.
        
        if found:
            return best_set

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Run the search
    result = _find_max_dissociated(deadline)
    
    # Ensure result is not empty
    if not result:
        result = [1]
        
    return result
