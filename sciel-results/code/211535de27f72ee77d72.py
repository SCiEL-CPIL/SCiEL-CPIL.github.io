import numpy as np
import random
import time

def construct():
    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 110  # leave some buffer
    
    counts = np.zeros(2001, dtype=np.int32) # Using numpy for speed
    
    # Try different orderings
    for seed in range(5000): # Increase trials
        if time.time() - start_time > time_limit:
            break
        
        random.seed(seed)
        candidates = list(range(1, 1001))
        
        # Strategy: Random shuffle is usually best for this heuristic
        random.shuffle(candidates)
        
        chosen = []
        chosen_arr = np.empty(0, dtype=np.int32)
        
        for x in candidates:
            # Check if adding x causes any sum to exceed 2
            # Sums are x + y for y in chosen, and 2*x
            if len(chosen) > 0:
                sums = x + chosen_arr
                # Check if any existing count for these sums is >= 2
                # Note: counts[sums] gets the counts for those specific sums
                if np.any(counts[sums] >= 2):
                    continue
            
            if counts[2 * x] >= 2:
                continue
            
            # Accept x
            chosen.append(x)
            if len(chosen) == 1:
                chosen_arr = np.array([x], dtype=np.int32)
            else:
                chosen_arr = np.append(chosen_arr, x)
            
            # Update counts
            if len(chosen) > 1:
                # Update for pairs (x, y) where y was in chosen before x
                # chosen_arr[:-1] are the previous elements
                counts[x + chosen_arr[:-1]] += 1
            
            counts[2 * x] += 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    return best_set
