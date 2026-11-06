import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Precompute all arithmetic progressions of length >= 2
    # An AP is defined by start (1-based) and step.
    # Indices in signs array are 0-based.
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = []
            # Generate terms: start, start+step, ... <= n
            cur = start
            while cur <= n:
                terms.append(cur - 1)
                cur += step
            if len(terms) >= 2:
                ap_indices.append(terms)
    
    # Note: The problem statement says "length including one". 
    # However, the sum of a length-1 AP is just the sign itself, which is always +/-1.
    # The max absolute sum will be at least 1.
    # If we include length 1, the max is at least 1.
    # If we exclude length 1, the max is determined by length >= 2.
    # Usually, "discrepancy" problems focus on non-trivial sums.
    # Let's check the contract: "any ... length including one".
    # So we must include length 1.
    # But since |sign[i]| = 1, the contribution to the max is 1.
    # If we find a solution with max sum 1 for all APs >= 2, the total max is 1.
    # If we find max sum 2 for APs >= 2, the total max is 2.
    # Including length 1 APs in the optimization loop is wasteful because their sum never changes magnitude (always 1) and doesn't affect other sums.
    # We can just check at the end: max(global_max_ap_sums, 1).
    # So we optimize for APs of length >= 2.

    num_aps = len(ap_indices)
    
    # Build position-to-AP mapping for fast updates
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
    
    # Convert pos_to_aps to numpy arrays for faster iteration?
    # Actually, list of lists is fine for indexing, but let's store as arrays if needed.
    # For now, lists are fast enough for the small number of affected APs per position.
    
    # Store AP indices as numpy arrays for vectorized sum computation during init/repair
    ap_arrays = [np.array(idx, dtype=np.int32) for idx in ap_indices]
    
    # Helper to compute all AP sums from signs
    def compute_ap_sums(signs):
        sums = np.empty(num_aps, dtype=np.int32)
        for k in range(num_aps):
            sums[k] = np.sum(signs[ap_arrays[k]])
        return sums

    # Helper to get max from histogram
    def get_max_from_hist(hist, current_max):
        # If current_max count is 0, scan down
        if hist[current_max] == 0:
            new_max = current_max - 1
            while new_max >= 0 and hist[new_max] == 0:
                new_max -= 1
            return new_max
        return current_max

    # Thue-Morse initialization
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    best_signs = thue_morse_signs(n)
    best_max = n + 1

    # Precompute pos_to_aps as lists of ints for fast access
    # We also need to know the delta for a flip.
    # If we flip sign[i] from s to -s, the sum of any AP containing i changes by -2*s.
    
    # Main Optimization Loop: Multi-restart SA with Greedy Repair
    # We will run several restarts or a long SA.
    # Given 85s, we can afford a long SA.
    
    # Let's use a single long SA run with strong greedy repair.
    signs = best_signs.copy()
    ap_sums = compute_ap_sums(signs)
    
    # Initialize histogram
    max_possible = n
    hist = np.zeros(max_possible + 1, dtype=np.int32)
    for s_val in ap_sums:
        hist[abs(s_val)] += 1
    
    current_max = max_possible
    while current_max > 0 and hist[current_max] == 0:
        current_max -= 1
        
    if current_max < best_max:
        best_max = current_max
        best_signs = signs.copy()

    temperature = 5.0
    cooling = 0.9998
    rng = random.Random(42)
    
    # Precompute affected APs for each position as numpy arrays for vectorized update?
    # No, we update ap_sums[affected] += delta. 
    # If affected is a list of ints, ap_sums[affected] works if affected is array.
    pos_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]

    it = 0
    while time.time() < deadline - 5.0:
        it += 1
        
        # Strategy:
        # 1. If we are stuck (high temp or just regular), do a random flip.
        # 2. If we have time, do a greedy repair step.
        
        # Greedy Repair: Find the worst AP and try to fix it.
        # This is more effective than random flips when we are close to the optimum.
        if current_max > 1 and rng.random() < 0.3:
            # Find an AP with max sum
            # We can sample a few worst APs to avoid O(N) scan every time?
            # Or just find one.
            # Since we have hist, we know current_max.
            # We need to find an index in ap_sums where abs(sum) == current_max.
            # This is O(N). N is ~3000. 3000 ops is fast.
            worst_aps = np.where(np.abs(ap_sums) == current_max)[0]
            if len(worst_aps) > 0:
                # Pick a random worst AP
                target_ap_id = worst_aps[rng.randint(0, len(worst_aps) - 1)]
                target_ap_arr = ap_arrays[target_ap_id]
                target_sum = ap_sums[target_ap_id]
                
                # Try flipping each position in this AP
                best_flip_idx = -1
                best_new_global_max = current_max
                best_new_ap_sum = abs(target_sum)
                
                for pos in target_ap_arr:
                    old_sign = signs[pos]
                    # Delta for this AP
                    new_ap_sum = target_sum - 2 * old_sign
                    
                    # Check if this flip reduces the global max
                    # We need to simulate the histogram update to be sure, 
                    # but we can approximate:
                    # If abs(new_ap_sum) < current_max, it might help.
                    # If abs(new_ap_sum) == current_max, it might be neutral or bad.
                    
                    # Let's do a full check for the best candidate.
                    # To save time, we only fully evaluate if it looks promising.
                    # If abs(new_ap_sum) >= current_max, it's unlikely to reduce global max
                    # unless it was the *only* AP at current_max and we reduce it?
                    # No, if we flip pos in target_ap, target_ap's sum changes.
                    # Other APs containing pos also change.
                    
                    # Let's just try the flip that minimizes abs(new_ap_sum) first.
                    # If multiple, pick one.
                    if abs(new_ap_sum) < best_new_ap_sum:
                        best_new_ap_sum = abs(new_ap_sum)
                        best_flip_idx = pos
                        best_new_global_max = current_max # Placeholder, will verify
                
                if best_flip_idx != -1:
                    pos = best_flip_idx
                    old_sign = signs[pos]
                    affected = pos_aps_arr[pos]
                    delta = -2 * old_sign
                    
                    # Apply flip
                    signs[pos] = -old_sign
                    old_abs_vals = np.abs(ap_sums[affected])
                    hist[old_abs_vals] -= 1
                    ap_sums[affected] += delta
                    new_abs_vals = np.abs(ap_sums[affected])
                    hist[new_abs_vals] += 1
                    
                    new_max = get_max_from_hist(hist, current_max)
                    # Check if new_max increased due to other affected APs
                    max_in_affected = np.max(new_abs_vals)
                    if max_in_affected > new_max:
                        new_max = max_in_affected
                    
                    if new_max < current_max:
                        current_max = new_max
                        if current_max < best_max:
                            best_max = current_max
                            best_signs = signs.copy()
                    elif new_max == current_max:
                        # Accept if it reduces the sum of the target AP? 
                        # We already picked the flip that reduced target AP sum.
                        # Accept to escape local minima? 
                        # Let's accept if it doesn't worsen global max.
                        current_max = new_max
                        if current_max < best_max:
                            best_max = current_max
                            best_signs = signs.copy()
                    else:
                        # Revert
                        signs[pos] = old_sign
                        ap_sums[affected] -= delta
                        hist[new_abs_vals] -= 1
                        hist[old_abs_vals] += 1
        else:
            # Random Flip with SA
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            affected = pos_aps_arr[i]
            delta = -2 * old_sign
            
            signs[i] = -old_sign
            old_abs_vals = np.abs(ap_sums[affected])
            hist[old_abs_vals] -= 1
            ap_sums[affected] += delta
            new_abs_vals = np.abs(ap_sums[affected])
            hist[new_abs_vals] += 1
            
            new_max = get_max_from_hist(hist, current_max)
            max_in_affected = np.max(new_abs_vals)
            if max_in_affected > new_max:
                new_max = max_in_affected
            
            delta_e = new_max - current_max
            
            if delta_e < 0 or (temperature > 0.01 and rng.random() < np.exp(-delta_e / temperature)):
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_sign
                ap_sums[affected] -= delta
                hist[new_abs_vals] -= 1
                hist[old_abs_vals] += 1
        
        temperature *= cooling
        
        # Periodic full restart if stuck?
        # If best_max hasn't improved in 10000 iters, maybe restart with different init?
        # For now, keep running.

    # Final Greedy Polish
    signs = best_signs.copy()
    ap_sums = compute_ap_sums(signs)
    hist = np.zeros(max_possible + 1, dtype=np.int32)
    for s_val in ap_sums:
        hist[abs(s_val)] += 1
    current_max = max_possible
    while current_max > 0 and hist[current_max] == 0:
        current_max -= 1
        
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        # Try flipping each position
        for i in range(n):
            old_sign = signs[i]
            affected = pos_aps_arr[i]
            delta = -2 * old_sign
            
            signs[i] = -old_sign
            old_abs_vals = np.abs(ap_sums[affected])
            hist[old_abs_vals] -= 1
            ap_sums[affected] += delta
            new_abs_vals = np.abs(ap_sums[affected])
            hist[new_abs_vals] += 1
            
            new_max = get_max_from_hist(hist, current_max)
            max_in_affected = np.max(new_abs_vals)
            if max_in_affected > new_max:
                new_max = max_in_affected
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                signs[i] = old_sign
                ap_sums[affected] -= delta
                hist[new_abs_vals] -= 1
                hist[old_abs_vals] += 1

    # Include length 1 APs check (always 1)
    final_max = max(current_max, 1)
    
    return [int(s) for s in best_signs]

# Test/Debug
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Length: {len(res)}")
    # Verify max sum
    n = 256
    signs = np.array(res)
    max_sum = 0
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            s = 0
            cur = start
            while cur <= n:
                s += signs[cur - 1]
                cur += step
            if abs(s) > max_sum:
                max_sum = abs(s)
    print(f"Max Sum: {max_sum}")
