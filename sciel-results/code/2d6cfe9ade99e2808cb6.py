import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

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

    # 2. Initialization Functions
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def random_init(N, rng):
        return np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    # 3. Core Optimization
    def optimize(s_init, time_budget, rng):
        s = s_init.copy()
        
        # Compute initial sums
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
            
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max
        
        # Precompute indices for shuffling
        indices_list = list(range(N))
        
        # We will use a two-phase approach:
        # Phase 1: Greedy descent to minimize max.
        # Phase 2: Simulated Annealing to escape local minima.
        
        # To be fast, we check violations against a target D.
        # We start with D = current_max - 1.
        
        target_D = current_max - 1
        if target_D < 0:
            target_D = 0
            
        T = 1.0 # Temperature for SA
        T_min = 0.01
        cooling = 0.95
        
        stuck_counter = 0
        max_stuck = 1000 # If no improvement in 1000 steps, cool down or restart phase
        
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                
                # Simulate flip
                # We need to check if any affected sum exceeds target_D
                # But if we are in SA mode, we might allow exceeding target_D (up to current_max)
                
                # Strategy:
                # 1. Try to satisfy |sum| <= target_D.
                # 2. If stuck, allow |sum| <= current_max (relaxation).
                
                # Let's define the "cost" of a state.
                # If we can reduce current_max, we do it.
                
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
                    # Accept it
                    sums[affected] = new_sums_affected
                    s[i] = new_val
                    stuck_counter = 0
                    
                    # Check if we can lower target_D further?
                    # Actually, if we satisfied target_D, maybe we can lower it?
                    # We can check if any sum is > target_D - 1?
                    # To keep it simple, we just keep satisfying target_D.
                    # When we get stuck (violates_target is always true for random flips),
                    # we will relax.
                    
                else:
                    # Move violates target_D but is within current_max.
                    # This is a "relaxed" move.
                    # Accept with probability based on how much it violates?
                    # Or just accept with some probability to explore.
                    
                    # Let's use a simple probability:
                    # If we have been stuck for a while, increase probability of accepting relaxed moves.
                    p_accept = 0.1 + 0.1 * (stuck_counter / max_stuck)
                    if rng.random() < p_accept:
                        sums[affected] = new_sums_affected
                        s[i] = new_val
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                
                # Check if we found a better solution
                # We need to know if the global max decreased.
                # This happens if all APs with |sum| == current_max are affected and reduced.
                # This is hard to check efficiently without a histogram.
                # Alternative: Periodically check the global max.
                
                # To avoid expensive global checks, we can check every 100 flips.
                # Or, we can maintain a count of max-valued APs.
                
                # Let's implement the count of max-valued APs for efficiency.
                # This requires updating a histogram or count variable.
                # Since we are rewriting, let's stick to the simpler "periodic check" first.
                # If it's too slow, we optimize.
                
                # Actually, let's do a cheap check:
                # If we accepted a "strict" move (not violates_target), 
                # it's possible the max dropped if target_D was current_max - 1 
                # and we just fixed the last violator.
                # But we don't know if it was the last one.
                
                # Let's just do a global check every 50 iterations.
                # 50 * 256 = 12800 flips per check.
                # Global check is O(num_aps) = 180k ops.
                # 180k / 50 = 3600 ops per flip amortized.
                # This is acceptable.
                
                if stuck_counter % 50 == 0:
                    global_max = int(np.max(np.abs(sums)))
                    if global_max < best_score:
                        best_score = global_max
                        best_s = s.copy()
                        # Update current_max and target_D
                        current_max = global_max
                        target_D = current_max - 1
                        if target_D < 0: target_D = 0
                        stuck_counter = 0
                        T = 1.0 # Reset temperature
                    else:
                        # If no improvement, cool down T
                        T *= cooling
                        if T < T_min:
                            T = T_min

        return best_s, best_score

    # 4. Main Loop with Restarts
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N # Max possible
    
    # Budget allocation
    # Run 1: Thue-Morse (40%)
    # Run 2: Random (30%)
    # Run 3: Random (30%)
    
    time_left = time_limit - (time.time() - start_time)
    
    # Run 1
    tm = thue_morse(N)
    s1, score1 = optimize(tm, time_left * 0.4, rng)
    if score1 < best_score_overall:
        best_score_overall = score1
        best_s_overall = s1
        
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 2
        rand1 = random_init(N, rng)
        s2, score2 = optimize(rand1, time_left * 0.5, rng)
        if score2 < best_score_overall:
            best_score_overall = score2
            best_s_overall = s2
            
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0:
        # Run 3
        rand2 = random_init(N, rng)
        s3, score3 = optimize(rand2, time_left * 0.8, rng)
        if score3 < best_score_overall:
            best_score_overall = score3
            best_s_overall = s3
            
    # Final Polish
    # If we have time, run a final tight local search on the best solution
    time_left = time_limit - (time.time() - start_time)
    if time_left > 0.5 and best_s_overall is not None:
         # Just run optimize on best_s_overall with strict constraints
         s_final, score_final = optimize(best_s_overall, time_left * 0.9, rng)
         if score_final < best_score_overall:
             best_score_overall = score_final
             best_s_overall = s_final

    return [int(x) for x in best_s_overall]
