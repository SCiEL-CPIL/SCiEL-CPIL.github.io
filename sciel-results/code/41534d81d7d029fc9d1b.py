import time
import random
import sys

def construct():
    N = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit

    # 1. Precompute all Arithmetic Progressions
    # An AP is defined by start a (1-indexed), step d, and length L >= 1.
    # Indices are 0-based: a-1, a-1+d, ..., a-1+(L-1)d.
    # We need to consider ALL such APs contained in [1, N].
    
    # Optimization: 
    # For a fixed step d, the APs starting at a are disjoint in their "start" but overlapping in indices.
    # However, the constraint is on ANY AP. 
    # Note: For d=1, these are just contiguous intervals.
    # For d>1, they are sparse.
    
    # Let's count how many APs there are.
    # d=1: N starts, lengths 1..N-a+1. Sum_{a=1}^N (N-a+1) = N(N+1)/2 ~ 32896
    # d=2: N starts, lengths 1..floor((N-a)/2)+1.
    # Total APs is roughly N^2/2 * log N? No, sum_{d=1}^N sum_{a=1}^N (N-a)/d + 1
    # ~ N^2/2 * H_N ~ 32768 * 5.8 ~ 190k.
    
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
            # We add all prefixes
            # Note: A prefix of length 1 is just the element itself.
            # This is included.
            for L in range(1, len(chain) + 1):
                ap_indices = chain[:L]
                ap_id = len(aps)
                aps.append(ap_indices)
                for idx in ap_indices:
                    aps_containing[idx].append(ap_id)
                    
    num_aps = len(aps)
    
    # 2. Initialization
    # Use Thue-Morse sequence as a strong initial guess.
    # Thue-Morse has low discrepancy on many structures.
    s = []
    for i in range(N):
        # Count bits in i
        bits = bin(i).count('1')
        s.append(1 if bits % 2 == 0 else -1)
    
    # Calculate initial sums
    ap_sums = [0] * num_aps
    for ap_id, ap_indices in enumerate(aps):
        total = 0
        for idx in ap_indices:
            total += s[idx]
        ap_sums[ap_id] = total
        
    # Histogram of absolute sums
    max_val = N
    hist = [0] * (max_val + 1)
    for total in ap_sums:
        hist[abs(total)] += 1
        
    current_max = max_val
    while current_max > 0 and hist[current_max] == 0:
        current_max -= 1
        
    best_max = current_max
    best_s = s[:]
    
    # 3. Local Search with Simulated Annealing
    # We use a hybrid approach:
    # - Greedy descent when possible (if new_max < current_max)
    # - Simulated Annealing to escape local minima
    
    rng = random.Random(42)
    
    T = 2.0
    T_min = 0.01
    alpha = 0.999
    
    # To speed up, we can precompute the list of APs containing each index as a tuple
    # But lists are fine for now.
    
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        i = rng.randrange(N)
        
        old_val = s[i]
        delta = -2 * old_val
        
        # Apply flip to sums and histogram
        revert_list = []
        max_new_abs = 0
        
        for ap_id in aps_containing[i]:
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
                best_s = s[:]
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
                # Use log for stability? No, delta_max is small.
                # exp(-1/0.01) is 0.
                # We can just check if rng.random() < exp(-delta_max/T)
                # But exp() is slow.
                # Since T is small, this is rarely accepted.
                # We can approximate or just use the formula.
                if delta_max <= 0:
                    accept = True
                else:
                    # Optimization: if T is very small, probability is 0
                    if T < 0.05 and delta_max > 1:
                        accept = False
                    else:
                        # Use math.exp
                        # import math
                        # accept = rng.random() < math.exp(-delta_max / T)
                        # Avoid import inside loop
                        # Simple approximation or just use pow
                        # 2.71828 ** (-delta_max / T)
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

    return best_s
