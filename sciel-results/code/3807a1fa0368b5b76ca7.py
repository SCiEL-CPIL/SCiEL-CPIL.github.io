import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute APs
    # An AP is defined by start and step. Length must be >= 2 to be interesting for max sum > 1.
    # However, the problem says "including one". Max abs sum for length 1 is 1.
    # So if we find a config with max sum 1, it's optimal.
    # We will include all APs with length >= 2 in our optimization, 
    # and the final check will naturally include length 1 (which is always 1).
    
    aps_indices = []
    # aps_containing[i] will be a list of indices into aps_indices
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_id = len(aps_indices)
                aps_indices.append(np.array(terms, dtype=np.int32))
                for idx in terms:
                    aps_containing[idx].append(ap_id)
    
    # Convert aps_containing to numpy arrays for fast indexing
    aps_containing_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in aps_containing]
    
    num_aps = len(aps_indices)
    
    # 2. Helper functions
    
    def compute_ap_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int16)
        for ap_idx, indices in enumerate(aps_indices):
            sums[ap_idx] = np.sum(signs[indices])
        return sums

    def get_max_abs(sums):
        return np.max(np.abs(sums))

    # 3. Greedy Initialization
    def greedy_init():
        signs = np.zeros(N, dtype=np.int8)
        # We need to track sums efficiently.
        # Since we are building it, we can just recompute or maintain.
        # For N=256, recomputing is O(N * AvgLen) which is small.
        # But let's maintain sums.
        ap_sums = np.zeros(num_aps, dtype=np.int16)
        
        for i in range(N):
            best_sign = 1
            best_max = N + 1
            best_sums = None
            
            for s in [1, -1]:
                # Calculate new sums
                new_sums = ap_sums.copy()
                # Update sums for APs containing i
                ap_ids = aps_containing_np[i]
                if len(ap_ids) > 0:
                    new_sums[ap_ids] += s
                
                m = get_max_abs(new_sums)
                if m < best_max:
                    best_max = m
                    best_sign = s
                    best_sums = new_sums
            
            signs[i] = best_sign
            ap_sums = best_sums
            
        return signs, ap_sums, get_max_abs(ap_sums)

    # 4. Simulated Annealing / Local Search
    def local_search(signs, ap_sums, current_max, time_limit):
        best_signs = signs.copy()
        best_max = current_max
        
        T = 10.0  # Initial temperature
        cool_rate = 0.995
        min_T = 0.1
        
        # Precompute random indices for speed? No, just use random.randint
        
        while time.time() < time_limit and T > min_T:
            # Pick a random index to flip
            i = random.randint(0, N - 1)
            
            # Calculate delta
            # If signs[i] is 1, new is -1, change is -2
            # If signs[i] is -1, new is 1, change is +2
            delta = -2 * signs[i]
            
            # Get affected APs
            ap_ids = aps_containing_np[i]
            
            if len(ap_ids) == 0:
                # No APs affected, flip is free (doesn't change max)
                # But we should still update signs? Yes.
                signs[i] = -signs[i]
                # ap_sums unchanged
                # current_max unchanged
                # Continue
                continue
            
            # Apply change
            old_sums = ap_sums[ap_ids].copy()
            ap_sums[ap_ids] += delta
            
            new_max = get_max_abs(ap_sums)
            
            # Acceptance criterion
            accepted = False
            if new_max < current_max:
                accepted = True
            elif new_max == current_max:
                # Accept to move around plateau
                if random.random() < 0.1:
                    accepted = True
            else:
                # new_max > current_max
                # Delta E = new_max - current_max
                delta_e = new_max - current_max
                if random.random() < np.exp(-delta_e / T):
                    accepted = True
            
            if accepted:
                signs[i] = -signs[i]
                current_max = new_max
                
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                ap_sums[ap_ids] = old_sums
            
            T *= cool_rate
            
        return best_signs, best_max

    # 5. Main Loop
    best_val = N + 1
    best_signs = None
    
    # Run multiple restarts or long runs
    while time.time() < deadline - 5.0:
        # Greedy Init
        signs, ap_sums, current_max = greedy_init()
        
        if current_max < best_val:
            best_val = current_max
            best_signs = signs.copy()
            
        if best_val <= 2:
            break
            
        # Local Search
        # Allocate remaining time, leaving a buffer
        time_left = deadline - 5.0 - time.time()
        if time_left <= 0:
            break
            
        # Run SA for a portion of the time, e.g., 50% of remaining
        sa_time = max(1.0, time_left * 0.5)
        signs, best_max_ls = local_search(signs, ap_sums, current_max, time.time() + sa_time)
        
        if best_max_ls < best_val:
            best_val = best_max_ls
            best_signs = signs.copy()
            
    return best_signs.tolist()
