import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    N = 256

    # 1. Precompute ALL APs and inverse mapping
    # AP defined by start s (1-indexed), step d.
    # Indices in 0-indexed: s-1, s-1+d, ...
    
    ap_indices = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            
            # Every prefix is an AP
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # Convert to numpy arrays for faster access
    # ap_indices: list of arrays
    ap_indices_np = [np.array(lst, dtype=np.int32) for lst in ap_indices]
    # aps_containing: list of arrays
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    # Precompute sum of lengths for each AP (useful for tie-breaking)
    ap_lengths = np.array([len(lst) for lst in ap_indices], dtype=np.int32)

    # 2. Initialization Functions
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def random_init(N, rng):
        return np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    def perturbed_init(base_s, rng, num_flips):
        """Create a perturbed version of base_s by flipping some positions"""
        s = base_s.copy()
        for _ in range(num_flips):
            i = rng.randint(0, N - 1)
            s[i] = -s[i]
        return s

    # 3. Core Optimization with histogram tracking
    def optimize(s_init, time_budget, rng):
        s = s_init.copy()
        
        # Compute initial sums
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
            
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        
        # Build histogram of |sums| for O(1) max queries
        # Sums range from -N to N, so |sums| from 0 to N
        max_possible_sum = N  # Maximum absolute sum is N
        hist = np.zeros(max_possible_sum + 2, dtype=np.int32)
        for val in np.abs(sums):
            hist[val] += 1
        
        def get_max_from_hist():
            """Get current max from histogram"""
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
        
        # Precompute indices for shuffling
        indices_list = list(range(N))
        
        target_D = current_max - 1
        if target_D < 0:
            target_D = 0
            
        T = 1.0 # Temperature for SA
        T_min = 0.01
        cooling = 0.95
        
        stuck_counter = 0
        max_stuck = 500 # If no improvement in 500 steps, try perturbation
        
        perturbation_cooldown = 0
        
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                if perturbation_cooldown > 0:
                    perturbation_cooldown -= 1
                    continue
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                
                # Calculate new sums for affected
                new_sums_affected = sums[affected] + delta
                new_abs_affected = np.abs(new_sums_affected)
                
                # Check against target_D
                violates_target = np.any(new_abs_affected > target_D)
                
                # Check against current_max (hard limit for SA relaxation)
                violates_current_max = np.any(new_abs_affected > current_max)
                
                if violates_current_max:
                    # Never allow increasing the global max
                    continue
                
                if not violates_target:
                    # Move is valid for the stricter target
                    # Apply the flip
                    old_abs_affected = np.abs(sums[affected])
                    
                    # Update histogram
                    for j, ap_idx in enumerate(affected):
                        old_abs = old_abs_affected[j]
                        new_abs = new_abs_affected[j]
                        if old_abs != new_abs:
                            hist[old_abs] -= 1
                            hist[new_abs] += 1
                    
                    sums[affected] = new_sums_affected
                    s[i] = new_val
                    stuck_counter = 0
                    
                    # Check if we can lower target_D further
                    new_global_max = get_max_from_hist()
                    if new_global_max < current_max:
                        current_max = new_global_max
                        if current_max < best_score:
                            best_score = current_max
                            best_s = s.copy()
                        target_D = current_max - 1
                        if target_D < 0:
                            target_D = 0
                        T = 1.0 # Reset temperature
                        stuck_counter = 0
                    else:
                        # No improvement, but we made progress toward target
                        stuck_counter += 1
                        
                else:
                    # Move violates target_D but is within current_max.
                    # This is a "relaxed" move.
                    # Accept with probability based on how much it violates and stuck counter
                    max_violation = int(np.max(new_abs_affected - target_D))
                    p_accept = 0.05 + 0.15 * (stuck_counter / max_stuck) - 0.01 * max_violation
                    p_accept = max(0.0, min(0.5, p_accept))
                    if rng.random() < p_accept:
                        old_abs_affected = np.abs(sums[affected])
                        
                        # Update histogram
                        for j, ap_idx in enumerate(affected):
                            old_abs = old_abs_affected[j]
                            new_abs = new_abs_affected[j]
                            if old_abs != new_abs:
                                hist[old_abs] -= 1
                                hist[new_abs] += 1
                        
                        sums[affected] = new_sums_affected
                        s[i] = new_val
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                
                # If stuck for too long, try perturbation
                if stuck_counter >= max_stuck:
                    # Try a perturbation: flip 2-5 random positions
                    num_perturb = rng.randint(2, 5)
                    perturbation_indices = rng.sample(range(N), num_perturb)
                    
                    # Save state
                    s_backup = s.copy()
                    sums_backup = sums.copy()
                    hist_backup = hist.copy()
                    
                    # Apply perturbation
                    for pi in perturbation_indices:
                        old_val = s[pi]
                        delta = -2 * old_val
                        new_val = -old_val
                        affected = aps_containing_np[pi]
                        new_sums_affected = sums[affected] + delta
                        new_abs_affected = np.abs(new_sums_affected)
                        
                        # Only accept if it doesn't increase max
                        if not np.any(new_abs_affected > current_max):
                            old_abs_affected = np.abs(sums[affected])
                            for j, ap_idx in enumerate(affected):
                                old_abs = old_abs_affected[j]
                                new_abs = new_abs_affected[j]
                                if old_abs != new_abs:
                                    hist[old_abs] -= 1
                                    hist[new_abs] += 1
                            sums[affected] = new_sums_affected
                            s[pi] = new_val
                    
                    # Check if perturbation helped
                    new_global_max = get_max_from_hist()
                    if new_global_max < current_max:
                        current_max = new_global_max
                        if current_max < best_score:
                            best_score = current_max
                            best_s = s.copy()
                        target_D = current_max - 1
                        if target_D < 0:
                            target_D = 0
                        stuck_counter = 0
                        T = 1.0
                    else:
                        # Revert perturbation
                        s = s_backup
                        sums = sums_backup
                        hist = hist_backup
                        stuck_counter = 0
                        T *= cooling
                        if T < T_min:
                            T = T_min
                    perturbation_cooldown = 10

        return best_s, best_score

    # 4. Main Loop with Restarts
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N # Max possible
    
    # Budget allocation: more restarts with shorter individual runs
    # Run 1: Thue-Morse (30%)
    # Run 2: Random (20%)
    # Run 3: Perturbed Thue-Morse (20%)
    # Run 4: Random (20%)
    # Final: Polish best (10%)
    
    time_left = time_limit - (time.time() - start_time)
    
    # Run 1: Thue-Morse
    tm = thue_morse(N)
    s1, score1 = optimize(tm, time_left * 0.3, rng)
    if score1 < best_score_overall:
        best_score_overall = score1
        best_s_overall = s1
        
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 2: Random
        rand1 = random_init(N, rng)
        s2, score2 = optimize(rand1, time_left * 0.25, rng)
        if score2 < best_score_overall:
            best_score_overall = score2
            best_s_overall = s2
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 3: Perturbed Thue-Morse
        pert_tm = perturbed_init(tm, rng, 5)
        s3, score3 = optimize(pert_tm, time_left * 0.25, rng)
        if score3 < best_score_overall:
            best_score_overall = score3
            best_s_overall = s3
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 4: Random
        rand2 = random_init(N, rng)
        s4, score4 = optimize(rand2, time_left * 0.4, rng)
        if score4 < best_score_overall:
            best_score_overall = score4
            best_s_overall = s4
            
    # Final Polish
    # If we have time, run a final tight local search on the best solution
    time_left = time_limit - (time.time() - start_time)
    if time_left > 0.5 and best_s_overall is not None:
         # Run optimize on best_s_overall with strict constraints
         s_final, score_final = optimize(best_s_overall, time_left * 0.9, rng)
         if score_final < best_score_overall:
             best_score_overall = score_final
             best_s_overall = s_final

    return [int(x) for x in best_s_overall]
