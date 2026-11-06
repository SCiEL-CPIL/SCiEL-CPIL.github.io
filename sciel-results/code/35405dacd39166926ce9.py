import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 88.0  # Safety margin for 90s limit

    N = 256

    # 1. Precompute ALL APs
    # We need:
    # - ap_indices: list of lists of indices for each AP
    # - aps_ending_at[i]: list of AP indices that end at position i
    # - aps_containing[i]: list of AP indices that contain position i
    
    ap_indices = []
    aps_ending_at = [[] for _ in range(N)]
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            
            # All prefixes of base_indices are valid APs ending at various points
            # Actually, the task says "ANY arithmetic progression contained in [1,256]"
            # An AP is defined by start, step, and length.
            # So for a fixed start and step, every prefix is an AP.
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                
                # This AP ends at idx_list[-1]
                end_pos = idx_list[-1]
                aps_ending_at[end_pos].append(ap_idx)
                
                # This AP contains all indices in idx_list
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # Convert to numpy arrays for speed
    ap_indices_np = [np.array(lst, dtype=np.int32) for lst in ap_indices]
    aps_ending_at_np = [np.array(lst, dtype=np.int32) for lst in aps_ending_at]
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    # Precompute affected AP lengths for local search
    affected_lengths = np.array([len(lst) for lst in aps_containing], dtype=np.int32)

    # 2. Greedy Construction
    def greedy_construct(rng):
        s = np.zeros(N, dtype=np.int8)
        s[0] = 1
        
        # Current sums of all APs
        # We can maintain this incrementally.
        # Initially, only APs ending at 0 are active.
        sums = np.zeros(num_aps, dtype=np.int32)
        
        # Initialize sums for APs ending at 0
        # Only APs with start=1, step=d, length=1 end at 0? No, start=1 is index 0.
        # APs ending at 0:
        # Any AP with last element 0.
        # Since indices are 0..N-1, and APs are contiguous in index space? No, step can be > 1.
        # An AP is a set of indices {a, a+d, ...}.
        # If it ends at 0, then all previous elements must be < 0, so length must be 1.
        # So only APs of length 1 ending at 0.
        # These are singletons {0}.
        for ap_idx in aps_ending_at_np[0]:
            idx_list = ap_indices_np[ap_idx]
            sums[ap_idx] = s[idx_list[0]] # which is 1
        
        # Track max absolute sum
        current_max = 1
        hist = np.zeros(N + 2, dtype=np.int32)
        hist[1] = 1 # One AP with sum 1 (the singleton {0})
        # Wait, are there other APs ending at 0?
        # No, because indices are non-negative.
        
        for i in range(1, N):
            best_val = 0
            best_max = N + 1
            best_sums = None
            
            for val in [1, -1]:
                # Hypothetically set s[i] = val
                new_sums = sums.copy()
                max_abs = 0
                # Update sums for APs ending at i
                for ap_idx in aps_ending_at_np[i]:
                    idx_list = ap_indices_np[ap_idx]
                    # The new sum is old_sum + val (since i is the last element)
                    # Wait, does the AP sum include i? Yes.
                    # The previous sum was for the AP without i?
                    # No, `sums` holds the sums of *completed* APs.
                    # When we add position i, new APs ending at i are formed.
                    # These APs are prefixes of the sequences starting at some s.
                    # The sum of an AP ending at i is sum(s[a] for a in AP).
                    # We can compute this directly or incrementally.
                    # Incremental: sum(AP ending at i) = sum(AP ending at i-d) + s[i]
                    # But we don't store "AP ending at i-d" separately if it's not the full AP.
                    # Actually, every AP is a prefix of some maximal AP.
                    # Let's just compute the sum directly for the new APs.
                    # Since N is small (256) and num_aps is ~80k, but we only update ~O(N) APs per step?
                    # Number of APs ending at i is roughly sum_{d=1}^i (1) = i?
                    # For each step d, there is exactly 1 AP ending at i with that step?
                    # No. For a fixed step d, the APs ending at i are determined by the start.
                    # start = i - k*d.
                    # So for each d, there is 1 AP of length 1, 1 of length 2, etc?
                    # No, the AP is defined by start, step, length.
                    # If it ends at i, and has step d, then the elements are i, i-d, i-2d...
                    # So for a fixed i and d, there are multiple APs ending at i (different lengths).
                    # Specifically, length L such that i - (L-1)d >= 0.
                    # So yes, for each d, there are multiple APs ending at i.
                    
                    # To be fast, we can precompute the "parent" AP for each AP.
                    # An AP A ending at i with step d and length L > 1 has a parent AP A' ending at i-d with step d and length L-1.
                    # sum(A) = sum(A') + s[i].
                    
                    # Let's precompute parent pointers.
                    pass # We'll do this outside the loop for efficiency
                    
                # This direct computation inside the loop is too slow if not optimized.
                # Let's stick to the incremental sum update using parent pointers.
                
                break # Placeholder, real logic below
        
        return s

    # Optimization: Precompute parent pointers for APs
    # parent[ap_idx] = ap_idx of the AP that is a prefix of this one (length-1)
    # If length is 1, parent is -1.
    parent = np.full(num_aps, -1, dtype=np.int32)
    # We need to map (start, step, length) to index?
    # Or just iterate through APs and find parents.
    # Since APs are generated in a specific order, we can store (start, step, length) for each AP.
    ap_params = []
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            curr = s - 1
            while curr < N:
                # This is the start of a maximal AP? No, this is the start of a base sequence.
                # The APs are prefixes.
                # Let's re-generate and store params.
                break
            # Re-do generation to store params
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            for L in range(1, len(base_indices) + 1):
                ap_params.append((s-1, d, L))
                curr = s - 1
                break # Just to get out of inner loop quickly for logic check, actual loop below

    # Correct generation with params
    ap_params = []
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            for L in range(1, len(base_indices) + 1):
                ap_params.append((s-1, d, L))
                
    # Now map (start, step, length) to index
    # We can just iterate and set parent
    # But we need to find the index of the parent AP.
    # Parent of (start, step, L) is (start, step, L-1).
    # Let's create a dict for O(1) lookup.
    ap_to_idx = {}
    for idx, (st, stp, ln) in enumerate(ap_params):
        ap_to_idx[(st, stp, ln)] = idx
        
    for idx, (st, stp, ln) in enumerate(ap_params):
        if ln > 1:
            parent_idx = ap_to_idx.get((st, stp, ln-1), -1)
            parent[idx] = parent_idx
        else:
            parent[idx] = -1

    # Now Greedy Construction is fast
    def greedy_construct(rng):
        s = np.zeros(N, dtype=np.int8)
        s[0] = 1
        sums = np.zeros(num_aps, dtype=np.int32)
        
        # Initialize sums for APs ending at 0
        for ap_idx in aps_ending_at_np[0]:
            # Length is 1, sum is s[0] = 1
            sums[ap_idx] = 1
            
        hist = np.zeros(N + 2, dtype=np.int32)
        hist[1] = len(aps_ending_at_np[0])
        current_max = 1
        
        for i in range(1, N):
            best_val = 0
            best_max = N + 1
            best_sums_state = None # Store sums if we need to rollback? No, we can compute diff.
            
            for val in [1, -1]:
                # Calculate new max
                # Only APs ending at i change.
                # Their new sums are sums[parent] + val.
                # Other sums remain same.
                # So new_max = max(current_max, max(|new_sums| for aps ending at i))
                
                max_new_sums = 0
                for ap_idx in aps_ending_at_np[i]:
                    p = parent[ap_idx]
                    if p != -1:
                        new_sum = sums[p] + val
                    else:
                        # Length 1 AP
                        new_sum = val
                    abs_sum = abs(new_sum)
                    if abs_sum > max_new_sums:
                        max_new_sums = abs_sum
                
                candidate_max = max(current_max, max_new_sums)
                
                if candidate_max < best_max:
                    best_max = candidate_max
                    best_val = val
                    # We don't need to store best_sums_state yet, just the val
                    # But we need to know which val to pick if tie.
                    # Let's collect all best vals
                    pass
                elif candidate_max == best_max:
                    # Tie break: random or alternate
                    if rng.random() < 0.5:
                        best_val = val
                        best_max = candidate_max

            # Apply best_val
            s[i] = best_val
            # Update sums
            for ap_idx in aps_ending_at_np[i]:
                p = parent[ap_idx]
                if p != -1:
                    sums[ap_idx] = sums[p] + best_val
                else:
                    sums[ap_idx] = best_val
                hist[abs(sums[ap_idx])] += 1
                # We don't decrement hist for old sums because these APs didn't exist before (or had 0 sum? No, they were not "active" in the sense of having a sum? 
                # Actually, `sums` array is size num_aps. Initially 0.
                # When an AP "completes" (ends at i), we set its sum.
                # Before that, its sum was 0.
                # So we should decrement hist[0] for each new AP?
                # Yes.
                
            # Decrement hist[0] for the number of APs ending at i
            num_new_aps = len(aps_ending_at_np[i])
            hist[0] -= num_new_aps
            
            # Update current_max
            # We can just recalculate from hist? Or track it.
            # Since we only added new APs, current_max can only increase or stay same.
            # But wait, we are minimizing the max.
            # The "current_max" in the greedy step was the max of *existing* sums.
            # Now we add new sums.
            # So new global max is max(old_global_max, max_new_sums).
            current_max = max(current_max, best_max) # Wait, best_max was calculated as max(current_max, max_new_sums)
            # So current_max is now best_max.

        return s, sums, current_max

    # 3. Local Search Polish
    def local_search_polish(s_init, sums_init, current_max_init, time_budget, rng):
        s = s_init.copy()
        sums = sums_init.copy()
        current_max = current_max_init
        best_s = s.copy()
        best_score = current_max
        
        hist = np.zeros(N + 2, dtype=np.int32)
        for val in np.abs(sums):
            hist[val] += 1
        
        def get_max_from_hist():
            for k in range(N, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
        
        indices_list = list(range(N))
        
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = int(s[i])
                delta = -2 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                if len(affected) == 0:
                    continue
                    
                new_sums_affected = sums[affected] + delta
                new_abs_affected = np.abs(new_sums_affected)
                
                # Check if any new sum exceeds current_max
                if np.any(new_abs_affected > current_max):
                    continue
                
                old_abs_affected = np.abs(sums[affected])
                
                # Update hist
                for j in range(len(affected)):
                    old_abs = old_abs_affected[j]
                    new_abs = new_abs_affected[j]
                    if old_abs != new_abs:
                        hist[old_abs] -= 1
                        hist[new_abs] += 1
                
                sums[affected] = new_sums_affected
                s[i] = new_val
                
                new_global_max = get_max_from_hist()
                if new_global_max < current_max:
                    current_max = new_global_max
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                else:
                    # Revert
                    s[i] = old_val
                    sums[affected] -= delta
                    for j in range(len(affected)):
                        old_abs = old_abs_affected[j]
                        new_abs = new_abs_affected[j]
                        if old_abs != new_abs:
                            hist[old_abs] -= 1
                            hist[new_abs] += 1

        return best_s, best_score

    # 4. Main Loop
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N
    
    time_left = time_limit - (time.time() - start_time)
    
    # Run multiple greedy constructions
    for run in range(5):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        s_greedy, sums_greedy, max_greedy = greedy_construct(rng)
        
        if best_s_overall is None or max_greedy < best_score_overall:
            best_s_overall = s_greedy
            best_score_overall = max_greedy
            best_sums_overall = sums_greedy
            
        # Polish
        time_left = time_limit - (time.time() - start_time)
        if time_left > 2.0:
            s_polished, score_polished = local_search_polish(s_greedy, sums_greedy, max_greedy, time_left * 0.5, rng)
            if score_polished < best_score_overall:
                best_score_overall = score_polished
                best_s_overall = s_polished

    # Final polish on best
    time_left = time_limit - (time.time() - start_time)
    if time_left > 1.0 and best_s_overall is not None:
        # Need to recompute sums for best_s_overall if we didn't store them
        # For simplicity, just run local search which recomputes sums? 
        # My local_search_polish takes sums_init. 
        # If best_s_overall came from a previous run, we might not have its sums.
        # Let's just recompute sums for the final best.
        sums_final = np.zeros(num_aps, dtype=np.int32)
        for r, idx_list in enumerate(ap_indices_np):
            sums_final[r] = np.sum(best_s_overall[idx_list])
        max_final = int(np.max(np.abs(sums_final)))
        
        s_final, score_final = local_search_polish(best_s_overall, sums_final, max_final, time_left * 0.8, rng)
        if score_final < best_score_overall:
            best_score_overall = score_final
            best_s_overall = s_final

    return [int(x) for x in best_s_overall]
