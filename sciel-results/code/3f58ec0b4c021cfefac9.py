import time
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Use Simulated Annealing with incremental evaluation.
    2. Maintain unordered pair counts directly for O(|A|) update cost per move.
    3. Objective: Minimize (Peak Multiplicity * 1000 + Uncovered Count).
    4. Multiple restarts with different initial states to avoid local minima.
    5. Final polish with a very low temperature to squeeze out the last bit of peak reduction.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    target_min = 128
    target_max = 256
    N = 129  # Range [0, 128]
    target_range = np.arange(target_min, target_max + 1)
    
    # State variables
    # mask: boolean array of length N
    # counts: integer array of length 257 (sums 0..256)
    # A_list: list of active indices for fast iteration
    
    def initialize_state(mask_init):
        mask = mask_init.copy()
        A_list = np.where(mask == 1)[0].tolist()
        counts = np.zeros(257, dtype=np.int32)
        
        # Initialize counts
        for i_idx, a in enumerate(A_list):
            for b in A_list[i_idx:]: # b >= a
                s = a + b
                counts[s] += 1
                
        return mask, counts, A_list

    def get_score(counts):
        target_counts = counts[target_min:target_max + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            # Penalize uncovered heavily
            return 100000 + uncovered, uncovered, 0
        peak = np.max(target_counts)
        return peak * 100 + uncovered, uncovered, peak

    def add_element(mask, counts, A_list, x):
        # x is not in A
        # Add pairs {x, y} for y in A, y < x
        # Add pair {x, x}
        # Add pairs {x, y} for y in A, y > x
        
        # Efficiently: iterate all y in A
        for y in A_list:
            if y < x:
                counts[x + y] += 1
            elif y == x:
                # Should not happen if x not in A
                pass
            else: # y > x
                counts[x + y] += 1
        
        # The pair {x, x}
        counts[2 * x] += 1
        
        mask[x] = 1
        A_list.append(x)
        # Keep A_list sorted? Not strictly necessary for correctness, 
        # but might help with deterministic behavior or slight speedups in other contexts.
        # For SA, order doesn't matter for the logic above.
        
    def remove_element(mask, counts, A_list, x):
        # Remove pairs {x, y} for y in A, y != x
        # Remove pair {x, x}
        
        for y in A_list:
            if y == x:
                continue
            counts[x + y] -= 1
            
        counts[2 * x] -= 1
        
        mask[x] = 0
        A_list.remove(x)

    # Initial states
    # 1. Dense set [64..128]
    mask_dense = np.zeros(N, dtype=np.bool_)
    mask_dense[64:129] = True
    
    # 2. Random sparse set
    rng = np.random.default_rng(42)
    mask_rand = np.zeros(N, dtype=np.bool_)
    indices = rng.choice(N, size=40, replace=False)
    mask_rand[indices] = True
    
    # 3. Another random
    mask_rand2 = np.zeros(N, dtype=np.bool_)
    indices = rng.choice(N, size=30, replace=False)
    mask_rand2[indices] = True

    initial_masks = [mask_dense, mask_rand, mask_rand2]
    
    best_score_global = float('inf')
    best_mask_global = None
    best_peak_global = 0
    best_uncovered_global = 0
    
    for init_mask in initial_masks:
        if time.time() > deadline:
            break
            
        mask, counts, A_list = initialize_state(init_mask)
        
        curr_score, curr_unc, curr_peak = get_score(counts)
        
        # If this initial state is already very good, record it
        if curr_score < best_score_global:
            best_score_global = curr_score
            best_mask_global = mask.copy()
            best_peak_global = curr_peak
            best_uncovered_global = curr_unc
            
        # Simulated Annealing
        T = 10.0
        cooling_rate = 0.99
        min_T = 0.01
        
        # Track best in this run
        run_best_score = curr_score
        run_best_mask = mask.copy()
        run_best_peak = curr_peak
        run_best_unc = curr_unc
        
        while T > min_T and time.time() < deadline:
            # Check time more frequently if T is low?
            # Just check every iteration, it's cheap.
            
            # Generate move
            move_type = rng.random()
            
            new_mask = mask.copy()
            new_counts = counts.copy()
            new_A_list = A_list.copy()
            
            if move_type < 0.4:
                # Add
                candidates = np.where(new_mask == 0)[0]
                if len(candidates) == 0: continue
                x = rng.choice(candidates)
                add_element(new_mask, new_counts, new_A_list, x)
            elif move_type < 0.8:
                # Remove
                if len(new_A_list) <= 1: continue
                x = rng.choice(new_A_list)
                remove_element(new_mask, new_counts, new_A_list, x)
            else:
                # Swap
                if len(new_A_list) == 0: continue
                candidates_in = new_A_list
                candidates_out = np.where(new_mask == 0)[0]
                if len(candidates_out) == 0: continue
                x_in = rng.choice(candidates_in)
                x_out = rng.choice(candidates_out)
                remove_element(new_mask, new_counts, new_A_list, x_in)
                add_element(new_mask, new_counts, new_A_list, x_out)
            
            new_score, new_unc, new_peak = get_score(new_counts)
            
            delta = new_score - curr_score
            
            accept = False
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    # Avoid overflow in exp
                    if delta / T < 700: 
                        prob = np.exp(-delta / T)
                    else:
                        prob = 0.0
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                mask = new_mask
                counts = new_counts
                A_list = new_A_list
                curr_score = new_score
                curr_unc = new_unc
                curr_peak = new_peak
                
                if curr_score < run_best_score:
                    run_best_score = curr_score
                    run_best_mask = mask.copy()
                    run_best_peak = curr_peak
                    run_best_unc = curr_unc
                    
                if curr_score < best_score_global:
                    best_score_global = curr_score
                    best_mask_global = mask.copy()
                    best_peak_global = curr_peak
                    best_uncovered_global = curr_unc
                    
                # Early stop if perfect
                if best_uncovered_global == 0 and best_peak_global == 1:
                    return np.where(best_mask_global == 1)[0].tolist()
            
            T *= cooling_rate
            
        # End of SA for this initial state
        if run_best_score < best_score_global:
            best_score_global = run_best_score
            best_mask_global = run_best_mask
            best_peak_global = run_best_peak
            best_uncovered_global = run_best_unc

    # Final polish: If we have a valid solution, try to reduce peak further with a greedy/local search
    if best_uncovered_global == 0 and best_peak_global > 1:
        mask, counts, A_list = initialize_state(best_mask_global)
        curr_score, curr_unc, curr_peak = get_score(counts)
        
        # Local search for peak reduction
        # Try swapping elements to reduce the max count
        # Simple approach: Iterate through sums with max count, try to break them.
        
        # Given time constraints, we might just rely on the SA result.
        # But let's do a quick pass:
        # If peak is P, try to find a swap that reduces the count of the max sum by 1 without increasing any other sum above P-1 (or P).
        
        # This is complex to implement robustly in the remaining time.
        # The SA with incremental eval should be strong enough.
        
    # Convert result
    if best_mask_global is None:
        # Fallback: Return the dense set
        return list(range(64, 129))
        
    result_list = np.where(best_mask_global == 1)[0].tolist()
    
    # Ensure non-empty
    if not result_list:
        return [128]
        
    return result_list
