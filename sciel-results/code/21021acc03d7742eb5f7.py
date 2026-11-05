import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    num_targets = max_sum - min_sum + 1  # 129
    
    # Objective: Minimize (Uncovered_Count * LARGE_WEIGHT + Peak_Multiplicity)
    # We want Uncovered_Count == 0.
    # Among valid solutions, minimize Peak_Multiplicity.
    
    def evaluate(A_mask):
        """
        A_mask: boolean array of size n+1 (indices 0..128)
        Returns: (uncovered_count, peak_multiplicity, total_pairs_in_range)
        """
        # Get indices of active elements
        active_indices = np.where(A_mask)[0]
        if len(active_indices) == 0:
            return num_targets, 0, 0
        
        # Calculate sums
        # To avoid O(m^2) loop in Python, we can use numpy broadcasting for small m
        # m is typically < 100. 100x100 = 10k ops. Very fast.
        # active_indices is shape (m,)
        # sums = active_indices[:, None] + active_indices[None, :]
        # This creates (m, m) matrix.
        # We only need upper triangle including diagonal (unordered pairs with repetition).
        
        # However, for very small m, Python loops might be faster due to overhead of numpy creation?
        # Let's stick to numpy for speed if m > 10.
        
        m = len(active_indices)
        if m < 5:
            counts = np.zeros(max_sum + 1, dtype=np.int32)
            for i in range(m):
                ai = active_indices[i]
                if min_sum <= 2*ai <= max_sum:
                    counts[2*ai] += 1
                for j in range(i+1, m):
                    aj = active_indices[j]
                    s = ai + aj
                    if min_sum <= s <= max_sum:
                        counts[s] += 1
        else:
            # Vectorized approach
            # Generate all pairs i <= j
            # indices_i, indices_j = np.triu_indices(m)
            # This creates large arrays. 
            # Alternative: 
            # sums = active_indices[:, None] + active_indices[None, :]
            # mask = np.triu(np.ones((m, m), dtype=bool))
            # valid_sums = sums[mask]
            # Then bincount.
            
            sums = active_indices[:, None] + active_indices[None, :]
            mask = np.triu(np.ones((m, m), dtype=bool))
            valid_sums = sums[mask]
            
            # Filter to target range [128, 256]
            valid_sums = valid_sums[(valid_sums >= min_sum) & (valid_sums <= max_sum)]
            
            if len(valid_sums) == 0:
                return num_targets, 0, 0
            
            # Count occurrences
            # bincount is fast
            counts_arr = np.bincount(valid_sums, minlength=max_sum + 1)
            counts = counts_arr[min_sum:max_sum+1]
            
        uncovered = np.sum(counts == 0)
        peak = np.max(counts) if len(counts) > 0 else 0
        
        return int(uncovered), int(peak), int(np.sum(counts))

    def get_objective(uncovered, peak):
        # Heavily penalize uncovered sums
        # If uncovered > 0, objective is large.
        # We want to minimize uncovered first, then peak.
        # Objective = uncovered * 1000 + peak
        return uncovered * 1000 + peak

    # Initialize
    # Start with a dense random set or a structured set.
    # A structured set like [64, 128] might be a good start?
    # Or just random 50% density.
    
    current_A = np.zeros(n + 1, dtype=bool)
    # Random initialization: pick ~40 elements
    init_size = 40
    init_indices = random.sample(range(n + 1), init_size)
    for idx in init_indices:
        current_A[idx] = True
    
    cur_unc, cur_peak, _ = evaluate(current_A)
    cur_obj = get_objective(cur_unc, cur_peak)
    
    best_A = current_A.copy()
    best_unc, best_peak, _ = evaluate(best_A)
    best_obj = get_objective(best_unc, best_peak)
    
    # Simulated Annealing Parameters
    T = 50.0  # Initial temperature
    T_min = 0.1
    alpha = 0.995  # Cooling rate
    # Number of iterations per temperature? Or just run until time runs out.
    # Let's run until time runs out, adjusting T.
    
    # To allow exploration, we occasionally do swaps instead of flips.
    # Flip: toggle one bit.
    # Swap: remove one, add another.
    
    # We need to track the set of active indices for fast swaps?
    # Or just use the mask.
    
    while time.time() < deadline - 2.0:
        # Generate neighbor
        if random.random() < 0.5:
            # Flip
            idx = random.randint(0, n)
            new_A = current_A.copy()
            new_A[idx] = not new_A[idx]
        else:
            # Swap: pick an active idx to remove and an inactive idx to add
            active_indices = np.where(current_A)[0]
            if len(active_indices) == 0:
                # Just flip
                idx = random.randint(0, n)
                new_A = current_A.copy()
                new_A[idx] = True
            else:
                remove_idx = active_indices[random.randint(0, len(active_indices)-1)]
                # Find inactive indices
                inactive_indices = np.where(~current_A)[0]
                if len(inactive_indices) == 0:
                    # Full set, can't swap. Just flip? No, if full, all sums covered?
                    # If full, peak is high. Maybe remove one.
                    new_A = current_A.copy()
                    new_A[remove_idx] = False
                else:
                    add_idx = inactive_indices[random.randint(0, len(inactive_indices)-1)]
                    new_A = current_A.copy()
                    new_A[remove_idx] = False
                    new_A[add_idx] = True
        
        new_unc, new_peak, _ = evaluate(new_A)
        new_obj = get_objective(new_unc, new_peak)
        
        # Acceptance probability
        if new_obj < cur_obj:
            # Accept
            current_A = new_A
            cur_unc, cur_peak = new_unc, new_peak
            cur_obj = new_obj
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_unc, best_peak = new_unc, new_peak
                best_A = current_A.copy()
        else:
            # Metropolis acceptance
            delta = new_obj - cur_obj
            if delta < 0:
                # Should have been accepted above, but just in case
                current_A = new_A
                cur_unc, cur_peak = new_unc, new_peak
                cur_obj = new_obj
            else:
                # Probability exp(-delta / T)
                # If T is very small, this is ~0
                if T > T_min:
                    # Use log to avoid underflow? 
                    # delta is integer. T is float.
                    # If delta/T > 50, prob is ~0.
                    if delta / T < 50:
                        prob = np.exp(-delta / T)
                        if random.random() < prob:
                            current_A = new_A
                            cur_unc, cur_peak = new_unc, new_peak
                            cur_obj = new_obj
                            if new_obj < best_obj:
                                best_obj = new_obj
                                best_unc, best_peak = new_unc, new_peak
                                best_A = current_A.copy()
        
        # Cool down
        T *= alpha
        
        # If we have a valid solution (unc=0) and low peak, we can tighten search?
        # If best_unc == 0, we are minimizing peak.
        # If best_unc > 0, we are trying to cover.
        
        # Optional: If we've been stuck at the same best_obj for a while, 
        # we could restart with a different seed or higher T.
        # But simple SA usually works.
        
        # Check if we can break early if we find a theoretical lower bound?
        # Lower bound on peak: 
        # Let m be size of A. Pairs ~ m^2/2.
        # Sums in range: 129.
        # If we assume uniform distribution, avg multiplicity ~ m^2/258.
        # Peak >= Avg.
        # If m=30, avg ~ 4.5. Peak >= 5.
        # If m=20, avg ~ 1.5. Peak >= 2.
        # But geometry matters.
        
    # Final verification and extraction
    # best_A is a boolean mask
    final_indices = np.where(best_A)[0]
    final_list = [int(x) for x in final_indices]
    
    # Verify validity one last time
    if not final_list:
        return list(range(64, 129)) # Fallback
        
    counts = np.zeros(max_sum + 1, dtype=int)
    m = len(final_list)
    for i in range(m):
        ai = final_list[i]
        if min_sum <= 2*ai <= max_sum:
            counts[2*ai] += 1
        for j in range(i+1, m):
            s = ai + final_list[j]
            if min_sum <= s <= max_sum:
                counts[s] += 1
                
    # Check coverage
    valid = True
    for s in range(min_sum, max_sum + 1):
        if counts[s] == 0:
            valid = False
            break
            
    if not valid:
        # If SA failed to find a valid solution, fallback to a dense set
        # Dense set [64, 128] covers [128, 256]
        # Sums: 64+64=128, 128+128=256.
        # Is every sum in between covered?
        # 64+x where x in [64,128] -> [128, 192]
        # y+128 where y in [64,128] -> [192, 256]
        # Union covers [128, 256].
        return list(range(64, 129))
        
    return final_list

# EVOLVE-BLOCK-END
