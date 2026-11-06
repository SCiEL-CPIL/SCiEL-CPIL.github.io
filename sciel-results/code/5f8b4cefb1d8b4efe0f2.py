import time
import numpy as np
import random

def construct():
    start_time = time.time()
    DEADLINE = 58.0
    
    n = 128
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1
    
    def init_state(A_set):
        rep = np.zeros(NUM_SUMS, dtype=np.int32)
        ind = np.zeros(2*n + 1, dtype=np.bool_)
        for a in A_set:
            ind[a] = True
        # Calculate initial rep counts
        A_list = list(A_set)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if MIN_SUM <= s <= MAX_SUM:
                    rep[s - MIN_SUM] += 1
        return rep, ind

    def get_peak(rep):
        return int(np.max(rep))
    
    def is_valid(rep):
        return np.min(rep) >= 1

    best_peak = 999999
    best_A = set()
    
    # Generate diverse initial sets
    initial_sets = []
    # 1. The known good seed
    initial_sets.append(set(range(64, 129)))
    
    # 2. Random sets of various sizes
    sizes = [50, 55, 60, 65, 70, 75, 80]
    for _ in range(10):
        for size in sizes:
            if time.time() - start_time > DEADLINE:
                break
            rand_set = set(random.sample(range(0, 129), size))
            initial_sets.append(rand_set)
            
    # Process initial sets
    for init_set in initial_sets:
        if time.time() - start_time > DEADLINE:
            break
            
        rep, ind = init_state(init_set)
        if not is_valid(rep):
            continue
            
        A = set(init_set)
        current_peak = get_peak(rep)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = set(A)
            
        # Local Search
        improved = True
        while improved and (time.time() - start_time) < DEADLINE:
            improved = False
            
            # Try additions
            candidates_add = [x for x in range(0, 129) if not ind[x]]
            random.shuffle(candidates_add)
            
            for x in candidates_add:
                if time.time() - start_time > DEADLINE:
                    break
                
                new_rep = rep.copy()
                for y in A:
                    s = x + y
                    if MIN_SUM <= s <= MAX_SUM:
                        new_rep[s - MIN_SUM] += 1
                        
                if not is_valid(new_rep):
                    continue
                    
                new_peak = get_peak(new_rep)
                if new_peak < current_peak:
                    rep = new_rep
                    A.add(x)
                    ind[x] = True
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A)
                    break # Restart loop
            
            # Try removals
            if not improved:
                candidates_rem = list(A)
                random.shuffle(candidates_rem)
                
                for x in candidates_rem:
                    if time.time() - start_time > DEADLINE:
                        break
                    
                    new_rep = rep.copy()
                    for y in A:
                        if y == x:
                            continue
                        s = x + y
                        if MIN_SUM <= s <= MAX_SUM:
                            new_rep[s - MIN_SUM] -= 1
                    s_xx = 2 * x
                    if MIN_SUM <= s_xx <= MAX_SUM:
                        new_rep[s_xx - MIN_SUM] -= 1
                        
                    if not is_valid(new_rep):
                        continue
                    
                    new_peak = get_peak(new_rep)
                    if new_peak < current_peak:
                        rep = new_rep
                        A.remove(x)
                        ind[x] = False
                        current_peak = new_peak
                        improved = True
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = set(A)
                        break # Restart loop

    return list(best_A)
