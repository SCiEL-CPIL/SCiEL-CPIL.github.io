import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or numpy array of +1/-1.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    
    # Iterate over step d
    for d in range(1, n + 1):
        # For each residue class r in 0..d-1
        # We only care if the AP has at least 2 elements? 
        # The problem says "length including one". 
        # A singleton has sum +/-1, abs 1.
        # So max_abs is at least 1.
        for r in range(min(d, n)):
            # Extract subsequence: indices r, r+d, r+2d, ...
            # np.arange(r, n, d)
            # Using slicing is faster: signs_arr[r::d]
            sub = signs_arr[r::d]
            m = len(sub)
            if m <= 1:
                # Sum is +/-1, abs is 1
                if 1 > max_abs:
                    max_abs = 1
                continue
            
            # Calculate max subarray sum and min subarray sum
            # cumsum
            cum = np.cumsum(sub)
            
            # To find max subarray sum:
            # max_{i,j, i<j} (cum[j] - cum[i])
            # This is max(cum[j]) - min(cum[i]) where i < j.
            # We can use minimum.accumulate
            # Prepend 0 to cum to allow subarray starting at 0
            cum_ext = np.concatenate(([0], cum))
            
            min_prefix = np.minimum.accumulate(cum_ext)
            # min_prefix[i] is min(cum_ext[0..i])
            
            # Max subarray ending at j (1-indexed in cum_ext, 0-indexed in cum)
            # diff = cum[j] - min(cum[0..j-1])
            # Note: cum_ext has length m+1. cum has length m.
            # cum[j] corresponds to cum_ext[j+1].
            # We want max over j=0..m-1 of cum_ext[j+1] - min_prefix[j]
            
            # Vectorized:
            # diffs = cum_ext[1:] - min_prefix[:-1]
            # max_sub = np.max(diffs)
            
            # Similarly for min subarray:
            # min_{i<j} (cum[j] - cum[i]) = min(cum[j]) - max(cum[i])
            max_prefix = np.maximum.accumulate(cum_ext)
            # diffs_min = cum_ext[1:] - max_prefix[:-1]
            # min_sub = np.min(diffs_min)
            
            # Combine
            max_sub = np.max(cum_ext[1:] - min_prefix[:-1])
            min_sub = np.min(cum_ext[1:] - max_prefix[:-1])
            
            current_max = max(abs(max_sub), abs(min_sub))
            if current_max > max_abs:
                max_abs = current_max
                # Early exit? If max_abs is already high, can't exit easily 
                # unless we have a bound, but we are minimizing, so we need exact max.
                
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Thue-Morse
    tm = []
    for i in range(n):
        cnt = bin(i).count('1')
        tm.append(1 if cnt % 2 == 0 else -1)
    candidates.append(tm)
    
    # Walsh-Hadamard (Rademacher functions)
    # s_i = (-1)^{popcount(i & (i-1))}? No, simple Walsh.
    # Let's use a few frequency components.
    for k in [1, 2, 3, 4]:
        wh = []
        for i in range(n):
            # Simple bit check
            # (-1)^(bit k of i)
            wh.append(1 if (i >> k) & 1 == 0 else -1)
        candidates.append(wh)
        
    # Random
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng = random.Random(seed)
        rand = [rng.choice((-1, 1)) for _ in range(n)]
        candidates.append(rand)
        
    # Sine wave
    for freq in [0.5, 0.7, 1.0, 1.5, 2.0]:
        sine = [1 if math.sin(2 * math.pi * freq * i / n) >= 0 else -1 for i in range(n)]
        candidates.append(sine)

    # Evaluate and pick best
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 5:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]

    # 2. Local Search (Hill Climbing) - Optimized for speed
    signs = best_signs[:]
    current_disc = best_disc
    no_improve_count = 0
    max_no_improve = 10
    rng = random.Random(42)
    
    while time.time() < deadline - 3:
        improved = False
        
        # Try a limited number of random flips per iteration
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            
            idx = rng.randint(0, n - 1)
            old_val = signs[idx]
            signs[idx] = -old_val
            
            new_disc = _max_ap_discrepancy(signs)
            
            if new_disc < current_disc:
                current_disc = new_disc
                improved = True
                no_improve_count = 0
            else:
                signs[idx] = old_val
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Keep the flip to escape local minimum
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
                new_disc = _max_ap_discrepancy(signs)
                if new_disc < current_disc:
                    current_disc = new_disc
                    no_improve_count = 0
                else:
                    # Revert if much worse
                    if new_disc > best_disc:
                        signs[idx] = -signs[idx]
                    else:
                        current_disc = new_disc
                        no_improve_count = 0

    return signs
