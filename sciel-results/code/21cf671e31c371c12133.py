import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0
    
    # 1. Precompute APs
    # ap_list[ap_id] = list of indices
    # idx_to_aps[i] = list of ap_ids containing i
    ap_list = []
    idx_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        for start in range(n):
            # Generate AP: start, start+d, ... < n
            indices = []
            curr = start
            while curr < n:
                indices.append(curr)
                curr += d
            if len(indices) > 0:
                ap_id = len(ap_list)
                ap_list.append(indices)
                for idx in indices:
                    idx_to_aps[idx].append(ap_id)
    
    num_aps = len(ap_list)
    
    # 2. Initialize State
    rng = random.Random(42)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    
    # Initial sums
    sums = [0] * num_aps
    for ap_id, indices in enumerate(ap_list):
        s = 0
        for idx in indices:
            s += signs[idx]
        sums[ap_id] = s
        
    # Counts of sums
    # Range of sum: -n to n
    offset = n
    size = 2 * n + 1
    counts = [0] * size
    
    for s_val in sums:
        counts[s_val + offset] += 1
        
    # Find initial max discrepancy
    def get_max_disc():
        # Search from n down to 0
        for d_val in range(n, -1, -1):
            if d_val == 0:
                return 0
            if counts[d_val + offset] > 0 or counts[-d_val + offset] > 0:
                return d_val
        return 0
        
    current_max = get_max_disc()
    best_signs = signs[:]
    best_max = current_max
    
    # 3. Optimization Loop (Simulated Annealing)
    T = 5.0
    cooling = 0.995
    step = 0
    
    while time.time() - start_time < time_limit:
        # Pick random index
        i = rng.randint(0, n - 1)
        old_sign = signs[i]
        new_sign = -old_sign
        
        # Flip
        signs[i] = new_sign
        
        # Update sums and counts
        # Delta is 2 * new_sign (since old was -new_sign)
        # Actually, sum changes by 2 * new_sign?
        # old contribution: old_sign. new contribution: new_sign.
        # delta = new_sign - old_sign = new_sign - (-new_sign) = 2 * new_sign.
        delta = 2 * new_sign
        
        affected_aps = idx_to_aps[i]
        
        # We need to track if current_max changes
        # We will update counts, then re-evaluate max if necessary
        # To be efficient, we can just update counts and check if the bucket for current_max is empty.
        
        for ap_id in affected_aps:
            old_sum = sums[ap_id]
            new_sum = old_sum + delta
            
            # Update counts
            counts[old_sum + offset] -= 1
            counts[new_sum + offset] += 1
            
            sums[ap_id] = new_sum
            
        # Determine new max discrepancy
        # If the previous max bucket is empty, we need to find the new max.
        # It might have increased or decreased.
        # Simple approach:
        # 1. Check if current_max is still valid.
        # 2. If not, search down.
        # 3. But what if it increased?
        #    The new sums of affected APs could be larger than current_max.
        
        new_max = current_max
        
        # Check if any affected AP has sum > new_max
        # And if any affected AP made current_max invalid, we might need to drop.
        
        # Efficient check:
        # 1. If counts[new_max + offset] == 0 and counts[-new_max + offset] == 0:
        #      Then new_max must be lower.
        #      Search down from new_max - 1.
        # 2. Also, check if any new_sum has abs > new_max.
        
        max_abs_new = 0
        for ap_id in affected_aps:
            s_val = abs(sums[ap_id])
            if s_val > max_abs_new:
                max_abs_new = s_val
                
        # If max_abs_new > new_max, new_max increases.
        if max_abs_new > new_max:
            new_max = max_abs_new
        else:
            # Check if new_max is still supported
            if counts[new_max + offset] == 0 and counts[-new_max + offset] == 0:
                # Search down
                while new_max > 0 and counts[new_max + offset] == 0 and counts[-new_max + offset] == 0:
                    new_max -= 1
        
        delta_D = new_max - current_max
        
        # Acceptance
        if delta_D < 0 or rng.random() < math.exp(-delta_D / T):
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs[:]
        else:
            # Revert
            signs[i] = old_sign
            for ap_id in affected_aps:
                old_sum = sums[ap_id]
                new_sum = old_sum - delta # Revert delta
                
                counts[old_sum + offset] -= 1
                counts[new_sum + offset] += 1
                sums[ap_id] = new_sum
            # current_max remains unchanged
            
        T *= cooling
        step += 1
        
        # Safety check
        if step % 1000 == 0:
            if time.time() - start_time > time_limit:
                break
                
    return best_signs
