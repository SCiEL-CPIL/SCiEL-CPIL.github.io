import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Generate all APs
    ap_indices = []
    for start in range(n):
        for step in range(1, n):
            terms = []
            curr = start
            while curr < n:
                terms.append(curr)
                curr += step
            if len(terms) >= 1:
                ap_indices.append(terms)
                
    num_aps = len(ap_indices)
    
    # Precompute pos_to_aps
    temp_pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            temp_pos_to_aps[i].append(ap_id)
            
    pos_to_aps = [np.array(lst, dtype=np.int32) for lst in temp_pos_to_aps]

    # 2. Initialize State with Thue-Morse
    thue_morse = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        if bin(i).count('1') % 2 == 0:
            thue_morse[i] = 1
        else:
            thue_morse[i] = -1
            
    best_signs = thue_morse.copy()
    best_signs_copy = thue_morse.copy()
    
    # Compute initial sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for ap_id, idx in enumerate(ap_indices):
        ap_sums[ap_id] = np.sum(thue_morse[idx])
        
    # Compute initial histogram
    offset = 256
    hist_size = 513
    hist = np.zeros(hist_size, dtype=np.int32)
    np.add.at(hist, ap_sums + offset, 1)
    
    def get_max_abs_sum(h):
        for k in range(256, -1, -1):
            if h[offset + k] > 0 or h[offset - k] > 0:
                return k
        return 0
        
    current_max = get_max_abs_sum(hist)
    best_max = current_max
    
    # 3. Simulated Annealing
    rng = random.Random(42)
    temperature = 5.0
    cooling = 0.9995
    min_temp = 0.05
    
    moves = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        i = rng.randint(0, n - 1)
        old_sign = best_signs[i]
        new_sign = -old_sign
        delta = 2 * new_sign
        
        affected = pos_to_aps[i]
        
        old_sums = ap_sums[affected]
        new_sums = old_sums + delta
        
        hist -= np.bincount(old_sums + offset, minlength=hist_size)
        hist += np.bincount(new_sums + offset, minlength=hist_size)
        
        ap_sums[affected] = new_sums
        
        new_max = get_max_abs_sum(hist)
        
        delta_max = new_max - current_max
        accepted = False
        
        if new_max < current_max:
            accepted = True
        elif temperature > min_temp:
            if rng.random() < np.exp(-delta_max / temperature):
                accepted = True
                
        if accepted:
            best_signs[i] = new_sign
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs_copy = best_signs.copy()
                no_improve_count = 0
                if best_max == 1:
                    break
            else:
                no_improve_count += 1
        else:
            ap_sums[affected] = old_sums
            hist += np.bincount(old_sums + offset, minlength=hist_size)
            hist -= np.bincount(new_sums + offset, minlength=hist_size)
            no_improve_count += 1
            
        temperature *= cooling
        moves += 1
        
        # Optional: Restart if stuck
        if no_improve_count > 5000:
             # Perturb best solution
             for _ in range(10):
                 idx = rng.randint(0, n-1)
                 best_signs[idx] = -best_signs[idx]
             # Recompute sums and hist from scratch? Too slow.
             # Just reset temperature and continue?
             # Or restart SA from best_signs_copy?
             # Let's just reset temperature
             temperature = 2.0
             no_improve_count = 0
             
    return [int(s) for s in best_signs_copy]
