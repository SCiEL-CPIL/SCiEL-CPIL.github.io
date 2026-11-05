import numpy as np
import random
import time

def construct():
    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 110
    
    # Use NumPy for faster vectorized checks
    # counts array
    counts = np.zeros(2001, dtype=np.int8)
    
    for seed in range(2000): # More iterations if faster
        if time.time() - start_time > time_limit:
            break
        
        random.seed(seed)
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        chosen = []
        # We keep a numpy array of chosen elements for vectorization
        # But appending to numpy array is slow. 
        # We can maintain a list and convert to array for the check, 
        # or just use the list in the check if we don't vectorize.
        # Let's try vectorizing the check using the list converted to array.
        
        for x in candidates:
            if not chosen:
                chosen.append(x)
                counts[2*x] += 1
                continue
                
            # Check if x can be added
            # sums = x + chosen
            # We need to check if any count[sum] >= 2
            # And count[2x] >= 2
            
            # Convert chosen to numpy array for fast addition
            ch_arr = np.array(chosen)
            sums = x + ch_arr
            # Check max count in sums
            # counts is an array, so counts[sums] is vectorized
            if np.any(counts[sums] >= 2) or counts[2*x] >= 2:
                continue
                
            # Valid
            chosen.append(x)
            # Update counts
            counts[sums] += 1
            counts[2*x] += 1
            
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            
    return best_set
