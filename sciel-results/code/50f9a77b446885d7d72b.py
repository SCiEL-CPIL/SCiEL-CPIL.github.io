import random
import time

def construct():
    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 110  # leave some buffer
    
    for seed in range(2000):  # Try more seeds
        if time.time() - start_time > time_limit:
            break
        
        random.seed(seed)
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        chosen = set()
        counts = {}
        forbidden = set()
        
        for x in candidates:
            if x in forbidden:
                continue
            
            chosen.add(x)
            
            new_bad_sums = []
            
            # Iterate over chosen to update counts
            # Using a list copy might be faster if we were modifying chosen, but we are not.
            # However, iterating over a set is fine.
            for y in chosen:
                s = x + y
                if s > 2000:
                    continue
                
                counts[s] = counts.get(s, 0) + 1
                
                if counts[s] == 2:
                    new_bad_sums.append(s)
            
            for s in new_bad_sums:
                # Add forbidden values
                # s - z for z in chosen
                # Optimization: if s - z is out of bounds [1, 1000], we can skip adding to forbidden?
                # Yes, because candidates are only 1..1000.
                for z in chosen:
                    val = s - z
                    if 1 <= val <= 1000:
                        forbidden.add(val)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)
            
    return best_set
