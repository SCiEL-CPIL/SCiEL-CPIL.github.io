import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin

    N = 256

    # 1. Precompute ALL APs
    ap_indices = []
    aps_containing = [[] for _ in range(N)]

    # Iterate over all valid (start, step) pairs
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            
            # Each prefix is a valid AP
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # Convert to numpy arrays for faster indexing
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]

    # 2. Initialization: Thue-Morse sequence
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    # 3. Optimization using local search with histogram tracking
    def optimize(s_init, time_budget, rng):
        s = s_init.copy()
        
        # Compute initial sums for all APs
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(s[idx_list])
            
        abs_sums = np.abs(sums)
        current_max = int(np.max(abs_sums))
        
        # Maintain a count histogram of absolute sum values
        # Max possible absolute sum is N (all +1 or all -1)
        counts = np.bincount(abs_sums, minlength=N + 2)
        
        best_s = s.copy()
        best_score = current_max
        
        indices_list = list(range(N))
        
        # Main optimization loop
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                
                # Save old sums for affected APs
                old_sums = sums[affected].copy()
                
                # Apply flip to sums
                sums[affected] += delta
                
                # Check if this flip increases the global max
                max_new_affected = int(np.max(np.abs(sums[affected])))
                if max_new_affected > current_max:
                    # Revert
                    sums[affected] = old_sums
                    continue
                
                # Update histogram
                old_abs = np.abs(old_sums)
                new_abs = np.abs(sums[affected])
                
                changed_mask = old_abs != new_abs
                if np.any(changed_mask):
                    changed_old = old_abs[changed_mask]
                    changed_new = new_abs[changed_mask]
                    
                    # Update counts
                    np.add.at(counts, changed_old, -1)
                    np.add.at(counts, changed_new, 1)
                    
                    # Check if global max decreased
                    if counts[current_max] == 0:
                        new_max = current_max
                        while new_max > 0 and counts[new_max] == 0:
                            new_max -= 1
                        current_max = new_max
                        
                        if current_max < best_score:
                            best_score = current_max
                            best_s = s.copy()
                
                # Commit the flip
                s[i] = new_val
        
        return best_s, best_score

    rng = random.Random(42)
    
    # Run 1: Start from Thue-Morse
    tm = thue_morse(N)
    s1, score1 = optimize(tm, time_limit * 0.4, rng)
    best_s = s1
    best_score = score1

    # Run 2: Start from random
    if time.time() - start_time < time_limit * 0.7:
        rand_init = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        s2, score2 = optimize(rand_init, time_limit * 0.3, rng)
        if score2 < best_score:
            best_score = score2
            best_s = s2

    # Run 3: Refine the best solution found
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        s3, score3 = optimize(best_s, remaining - 0.5, rng)
        if score3 < best_score:
            best_score = score3
            best_s = s3

    return [int(x) for x in best_s]
