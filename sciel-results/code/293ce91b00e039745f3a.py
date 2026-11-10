import numpy as np
import time

def construct():
    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # We can also try to use a specific heuristic ordering
    # e.g. shuffle, but bias towards middle?
    # Let's just do random restarts.
    
    for seed in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(seed)
        candidates = np.arange(1, 1001)
        np.random.shuffle(candidates)
        
        S = []
        sum_counts = np.zeros(2001, dtype=np.int8)
        
        for x in candidates:
            if S:
                S_arr = np.array(S)
                potential_sums = x + S_arr
                two_x = 2 * x
                
                if sum_counts[two_x] >= 2:
                    continue
                if np.any(sum_counts[potential_sums] >= 2):
                    continue
                
                sum_counts[potential_sums] += 1
                sum_counts[two_x] += 1
                S.append(x)
            else:
                S.append(x)
                sum_counts[2*x] += 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return sorted(best_set)
