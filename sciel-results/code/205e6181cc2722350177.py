import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    N = 256

    # 1. Precompute ALL APs and build incidence structure
    ap_indices = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # Build incidence: for each position i, list of APs it belongs to
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    # Precompute: for each position, the count of APs it's in (for O(1) energy delta)
    # Energy = sum_r (sum_r)^2
    # If we flip position i from val v to -v, each AP r containing i changes by -2v
    # New sum_r = old_sum_r - 2v
    # Delta energy = sum_{r in aps[i]} [(sum_r - 2v)^2 - sum_r^2]
    #              = sum_{r in aps[i]} [-4v * sum_r + 4v^2]
    #              = -4v * (sum of sum_r over r in aps[i]) + 4v^2 * len(aps[i])
    # Since v^2 = 1: delta = -4v * S_i + 4 * len(aps[i])
    # where S_i = sum of current AP sums for APs containing i
    
    # 2. Initialization
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int32)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def random_init(N, rng):
        return np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int32)

    # 3. Core optimization using L2 energy with vectorized greedy/SA
    def optimize(s_init, time_budget, rng):
        s = s_init.astype(np.int32).copy()
        
        # Compute initial AP sums
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(s[idx_list])
        
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        
        # Precompute: for each position, the list of AP indices
        # and count
        ap_counts = np.array([len(aps_containing[i]) for i in range(N)], dtype=np.int32)
        
        # For energy: we track S_i = sum of sums over APs containing i
        # Update: when flipping position j, for each AP r containing j, sums[r] changes by delta
        # Then for each position i, S_i changes by delta * (1 if i in same APs as j... no)
        # Actually S_i = sum_{r: i in r} sums[r]
        # When we flip j: sums[r] += delta for all r containing j
        # So S_i += delta * (number of APs containing both i and j)
        # This is expensive to track. Instead, just recompute S_i on the fly or use a different approach.
        
        # Alternative: just compute the energy delta directly per flip in a vectorized batch.
        # For a single flip at position i:
        #   affected = aps_containing[i]
        #   delta_sums = -2 * s[i]  (scalar)
        #   new_sums_affected = sums[affected] + delta_sums
        #   energy_delta = sum(new_sums_affected^2) - sum(sums[affected]^2)
        #   = sum(2*delta_sums*sums[affected] + delta_sums^2)
        #   = 2*delta_sums * sum(sums[affected]) + len(affected) * delta_sums^2
        
        # For greedy: try all N positions, pick the one with most negative energy_delta
        # This is O(N * avg_ap_count) per sweep, which is manageable.
        
        indices_list = list(range(N))
        
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                if len(affected) == 0:
                    continue
                
                # Energy delta
                S_aff = np.sum(sums[affected].astype(np.int32))
                energy_delta = 2 * delta * S_aff + len(affected) * (delta * delta)
                
                # Also check max constraint: new max should not exceed best_score
                new_sums_aff = sums[affected] + delta
                new_max_aff = int(np.max(np.abs(new_sums_aff)))
                
                # Accept if energy decreases AND doesn't increase global max
                if energy_delta < 0 and new_max_aff <= best_score:
                    sums[affected] = new_sums_aff
                    s[i] = new_val
                
                # Periodic global check
                if i % 64 == 0:
                    global_max = int(np.max(np.abs(sums)))
                    if global_max < best_score:
                        best_score = global_max
                        best_s = s.copy()
        
        return best_s, best_score

    # 4. Final minimax polish: direct greedy to reduce max AP sum
    def minimax_polish(s_init, time_budget, rng):
        s = s_init.astype(np.int32).copy()
        
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(s[idx_list])
        
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        
        indices_list = list(range(N))
        
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            improved = False
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                if len(affected) == 0:
                    continue
                
                new_sums_aff = sums[affected] + delta
                new_max_aff = int(np.max(np.abs(new_sums_aff)))
                
                # Accept if it reduces the global max or keeps it same but reduces L2
                if new_max_aff < current_max:
                    sums[affected] = new_sums_aff
                    s[i] = new_val
                    current_max = new_max_aff
                    improved = True
                elif new_max_aff == current_max:
                    # Check if L2 energy decreases
                    old_l2 = np.sum(sums[affected].astype(np.int32) ** 2)
                    new_l2 = np.sum(new_sums_aff.astype(np.int32) ** 2)
                    if new_l2 < old_l2:
                        sums[affected] = new_sums_aff
                        s[i] = new_val
                        improved = True
            
            # Update best
            if current_max < best_score:
                best_score = current_max
                best_s = s.copy()
            
            if not improved and time.time() - start_time > time_budget * 0.8:
                break
        
        return best_s, best_score

    # 5. Main loop
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N
    
    time_left = time_limit - (time.time() - start_time)
    
    # Run 1: Thue-Morse + L2 optimize
    tm = thue_morse(N)
    s1, score1 = optimize(tm, time_left * 0.3, rng)
    if score1 < best_score_overall:
        best_score_overall = score1
        best_s_overall = s1
        
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 2: Random + L2 optimize
        rand1 = random_init(N, rng)
        s2, score2 = optimize(rand1, time_left * 0.3, rng)
        if score2 < best_score_overall:
            best_score_overall = score2
            best_s_overall = s2
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 3: Random + L2 optimize
        rand2 = random_init(N, rng)
        s3, score3 = optimize(rand2, time_left * 0.3, rng)
        if score3 < best_score_overall:
            best_score_overall = score3
            best_s_overall = s3
            
    # Final minimax polish
    time_left = time_limit - (time.time() - start_time)
    if time_left > 0.5 and best_s_overall is not None:
        s_final, score_final = minimax_polish(best_s_overall, time_left * 0.8, rng)
        if score_final < best_score_overall:
            best_score_overall = score_final
            best_s_overall = s_final

    return [int(x) for x in best_s_overall]
