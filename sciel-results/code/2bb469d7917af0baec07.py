import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute APs
    # An AP is defined by start and step. 
    # We only care about APs with length >= 2 for the optimization, 
    # as length 1 APs have sum +/-1, which is the theoretical minimum.
    # If we achieve max abs sum 1, it is optimal.
    
    aps_indices = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) > 1:
                ap_id = len(aps_indices)
                aps_indices.append(np.array(terms, dtype=np.int32))
                for idx in terms:
                    aps_containing[idx].append(ap_id)
    
    # Convert aps_containing to numpy arrays for fast indexing if needed, 
    # but lists are fine for iteration in the flip logic.
    num_aps = len(aps_indices)
    
    # Histogram parameters
    # Sums range from -N to N
    HIST_SIZE = 2 * N + 1
    OFFSET = N
    
    def init_histogram(ap_sums):
        hist = np.zeros(HIST_SIZE, dtype=np.int32)
        for s in ap_sums:
            hist[s + OFFSET] += 1
        return hist

    def get_max_from_hist(hist):
        # Scan from max possible absolute value down to 0
        for val in range(N, -1, -1):
            # Check positive val
            if hist[val + OFFSET] > 0:
                return val
            # Check negative val
            if val > 0 and hist[-val + OFFSET] > 0:
                return val
        return 0

    # 2. Greedy Initialization
    def greedy_init():
        signs = np.zeros(N, dtype=np.int8)
        ap_sums = np.zeros(num_aps, dtype=np.int16)
        hist = np.zeros(HIST_SIZE, dtype=np.int32)
        
        # Initially all AP sums are 0
        hist[0 + OFFSET] = num_aps
        current_max = 0
        
        for i in range(N):
            best_sign = 1
            best_max = N + 1
            best_ap_sums = None
            best_hist = None
            
            # Try sign 1
            # We need to simulate the change
            # To avoid copying the whole array, we can just calculate the new max
            # But we need to update the histogram.
            # For greedy init, N is small enough that we can just recompute or use a temporary hist.
            
            # Optimization: Since we are adding one sign at a time, 
            # the number of affected APs is small.
            
            for s in [1, -1]:
                # Create a copy of hist to modify
                temp_hist = hist.copy()
                temp_ap_sums = ap_sums.copy() # This is expensive. 
                
                # Better: Just calculate the new max directly from the affected APs?
                # No, the max might be determined by an unaffected AP.
                # But if the unaffected APs had max M, and the new sums are within [-M, M], 
                # then max is M.
                # However, we don't know M easily without scanning hist.
                
                # Let's stick to the copy for simplicity in init, as it's only N steps.
                # N=256, num_aps ~ 30k. Copying 30k int16s is fast.
                
                affected_aps = aps_containing[i]
                if len(affected_aps) > 0:
                    for ap_idx in affected_aps:
                        old_sum = temp_ap_sums[ap_idx]
                        new_sum = old_sum + s
                        temp_ap_sums[ap_idx] = new_sum
                        temp_hist[old_sum + OFFSET] -= 1
                        temp_hist[new_sum + OFFSET] += 1
                
                m = get_max_from_hist(temp_hist)
                
                if m < best_max:
                    best_max = m
                    best_sign = s
                    best_ap_sums = temp_ap_sums
                    best_hist = temp_hist
            
            signs[i] = best_sign
            ap_sums = best_ap_sums
            hist = best_hist
            current_max = best_max
            
        return signs, ap_sums, hist, current_max

    # 3. Simulated Annealing
    def local_search(signs, ap_sums, hist, current_max, time_limit):
        best_signs = signs.copy()
        best_max = current_max
        best_ap_sums = ap_sums.copy()
        best_hist = hist.copy()
        
        T = 10.0
        cool_rate = 0.995
        min_T = 0.1
        
        while time.time() < time_limit and T > min_T:
            i = random.randint(0, N - 1)
            
            # Delta for flipping sign[i]
            # If signs[i] is 1, new is -1, change is -2
            # If signs[i] is -1, new is 1, change is +2
            delta = -2 * signs[i]
            
            affected_aps = aps_containing[i]
            
            if len(affected_aps) == 0:
                # No change in sums, just flip sign? 
                # Actually, if no AP contains i (impossible for i in 0..N-1 with APs len>=2? 
                # Wait, singletons are ignored. But every index is in some AP of len>=2?
                # e.g., N=256. Index 0 is in AP (0,1), (0,2), etc.
                # Index 255 is in AP (255,1) - len 1? No, (255,1) is just [255]. 
                # APs must be contained in [0, N-1].
                # For d=1, start=255, terms=[255]. Len 1. Ignored.
                # So index 255 might not be in any AP of len>=2?
                # Let's check.
                # d=1: start=255 -> [255] (len 1)
                # d=2: start=255 -> [255] (len 1)
                # ...
                # So indices near the end might not be in any AP of len>=2.
                # In that case, flipping them doesn't change any AP sum.
                # So current_max stays the same.
                signs[i] = -signs[i]
                continue
            
            # Apply change
            # We need to update hist and ap_sums
            # And calculate new_max
            
            # To avoid copying, we can update in place and revert if rejected.
            
            old_sums = np.zeros(len(affected_aps), dtype=np.int16)
            for k, ap_idx in enumerate(affected_aps):
                old_sums[k] = ap_sums[ap_idx]
                hist[old_sums[k] + OFFSET] -= 1
                new_sum = old_sums[k] + delta
                ap_sums[ap_idx] = new_sum
                hist[new_sum + OFFSET] += 1
            
            new_max = get_max_from_hist(hist)
            
            accepted = False
            if new_max < current_max:
                accepted = True
            elif new_max == current_max:
                if random.random() < 0.1:
                    accepted = True
            else:
                delta_e = new_max - current_max
                if random.random() < np.exp(-delta_e / T):
                    accepted = True
            
            if accepted:
                signs[i] = -signs[i]
                current_max = new_max
                
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                    best_ap_sums = ap_sums.copy()
                    best_hist = hist.copy()
            else:
                # Revert
                for k, ap_idx in enumerate(affected_aps):
                    old_sum = old_sums[k]
                    new_sum = ap_sums[ap_idx] # This is the sum we just set
                    hist[old_sum + OFFSET] += 1
                    hist[new_sum + OFFSET] -= 1
                    ap_sums[ap_idx] = old_sum
            
            T *= cool_rate
        
        return best_signs, best_max

    # 4. Main Loop
    best_val = N + 1
    best_signs = None
    
    while time.time() < deadline - 5.0:
        signs, ap_sums, hist, current_max = greedy_init()
        
        if current_max < best_val:
            best_val = current_max
            best_signs = signs.copy()
            
        if best_val <= 2:
            break
            
        time_left = deadline - 5.0 - time.time()
        if time_left <= 0:
            break
            
        sa_time = max(1.0, time_left * 0.5)
        signs, best_max_ls = local_search(signs, ap_sums, hist, current_max, time.time() + sa_time)
        
        if best_max_ls < best_val:
            best_val = best_max_ls
            best_signs = signs.copy()
            
    return best_signs.tolist()
