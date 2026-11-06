import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    # 1. Precompute all Arithmetic Progressions (APs)
    # An AP is defined by start a (1-indexed), step d, and length L >= 1.
    # Indices are 0-based: a-1, a-1+d, ..., a-1+(L-1)d.
    
    # We store APs as lists of indices.
    # We also precompute for each index, which APs contain it.
    
    aps = []
    aps_containing = [[] for _ in range(N)]
    
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            start_idx = a - 1
            if start_idx >= N:
                continue
            
            # Generate the full chain for this start and step
            chain = []
            curr = start_idx
            while curr < N:
                chain.append(curr)
                curr += d
            
            # Every prefix of this chain is a valid AP
            for L in range(1, len(chain) + 1):
                ap_indices = chain[:L]
                ap_id = len(aps)
                aps.append(ap_indices)
                for idx in ap_indices:
                    aps_containing[idx].append(ap_id)
                    
    num_aps = len(aps)
    
    # 2. Initialization
    # Use Thue-Morse sequence as a strong initial guess.
    s = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
    
    # Calculate initial sums using numpy
    # For each AP, sum the signs at its indices
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for ap_id, ap_indices in enumerate(aps):
        ap_sums[ap_id] = np.sum(s[ap_indices])
        
    # Histogram of absolute sums
    max_val = N
    hist = np.zeros(max_val + 1, dtype=np.int32)
    abs_sums = np.abs(ap_sums)
    for val in abs_sums:
        hist[val] += 1
        
    current_max = max_val
    while current_max > 0 and hist[current_max] == 0:
        current_max -= 1
        
    best_max = current_max
    best_s = s.copy()
    
    # 3. Local Search with Simulated Annealing
    rng = random.Random(42)
    
    T = 2.0
    T_min = 0.01
    alpha = 0.999
    
    # Precompute aps_containing as lists of integers for faster access
    aps_containing_list = [list(x) for x in aps_containing]
    
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        i = rng.randrange(N)
        
        old_val = s[i]
        delta = -2 * old_val
        
        # Apply flip to sums and histogram
        revert_list = []
        max_new_abs = 0
        
        for ap_id in aps_containing_list[i]:
            old_sum = ap_sums[ap_id]
            new_sum = old_sum + delta
            
            old_abs = abs(old_sum)
            new_abs = abs(new_sum)
            
            if new_abs > max_new_abs:
                max_new_abs = new_abs
            
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            revert_list.append((ap_id, old_sum, old_abs, new_abs))
            ap_sums[ap_id] = new_sum
        
        # Determine new_max
        new_max = current_max
        
        if max_new_abs > new_max:
            new_max = max_new_abs
        else:
            # Check if current_max bucket is empty
            if hist[new_max] == 0:
                while new_max > 0 and hist[new_max] == 0:
                    new_max -= 1
        
        # Decision
        if new_max < current_max:
            # Always accept improvement
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_s = s.copy()
            s[i] = -old_val
        else:
            delta_max = new_max - current_max
            if delta_max == 0:
                # Accept with some probability to explore
                if rng.random() < 0.5:
                    s[i] = -old_val
                    current_max = new_max
                else:
                    # Revert
                    for ap_id, old_sum, old_abs, new_abs in revert_list:
                        ap_sums[ap_id] = old_sum
                        hist[old_abs] += 1
                        hist[new_abs] -= 1
            else:
                # delta_max > 0
                # Accept with prob exp(-delta_max / T)
                if T < 0.05 and delta_max > 1:
                    accept = False
                else:
                    accept = rng.random() < pow(2.71828, -delta_max / T)
                
                if accept:
                    s[i] = -old_val
                    current_max = new_max
                else:
                    # Revert
                    for ap_id, old_sum, old_abs, new_abs in revert_list:
                        ap_sums[ap_id] = old_sum
                        hist[old_abs] += 1
                        hist[new_abs] -= 1
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min

    return [int(x) for x in best_s]
