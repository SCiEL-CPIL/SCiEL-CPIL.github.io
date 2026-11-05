"""Direct heuristic construction for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list of +1/-1 indexed 0..n-1 (position i corresponds to index i+1).
    Returns the max absolute sum.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    max_abs = 0
    
    # Iterate over step d from 1 to n-1
    for d in range(1, n):
        # For each residue class mod d
        for r in range(d):
            if r >= n:
                continue
            
            # Get the subsequence at positions r, r+d, r+2d, ...
            # Length m
            m = (n - 1 - r) // d + 1
            if m <= 0:
                continue
            
            # Extract subsequence
            indices = np.arange(r, n, d)
            sub = signs_arr[indices]
            
            # Compute cumulative sum
            # Start with 0 to allow subarrays starting at the first element
            cum = np.concatenate(([0], np.cumsum(sub)))
            
            # Max subarray sum: max(C[j] - min(C[i] for i < j))
            # Min subarray sum: min(C[j] - max(C[i] for i < j))
            
            # Using NumPy for speed
            # running_min[i] = min(cum[0..i])
            running_min = np.minimum.accumulate(cum)
            running_max = np.maximum.accumulate(cum)
            
            # diff_max[j] = cum[j] - min(cum[0..j])
            # Note: The standard max subarray sum allows the subarray to start at 0.
            # The formula max_j (C[j] - min_{i<=j} C[i]) covers all subarrays ending at j.
            # Taking the max over j gives the global max subarray sum.
            
            diff_max = cum - running_min
            max_sum = np.max(diff_max)
            
            diff_min = cum - running_max
            min_sum = np.min(diff_min)
            
            current_max = max(abs(max_sum), abs(min_sum))
            if current_max > max_abs:
                max_abs = current_max
                
            # Early termination if we already found a large discrepancy?
            # Hard to prune without knowing bounds.
    
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    rng = random.Random(42)
    
    # Initialize with multiple strategies
    # Strategy 1: Thue-Morse (known to have low discrepancy for some structures)
    signs_tm = [0] * n
    for i in range(n):
        t = 0
        x = i
        while x:
            t ^= (x & 1)
            x >>= 1
        signs_tm[i] = 1 if t == 0 else -1
        
    # Strategy 2: Alternating
    signs_alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    
    # Strategy 3: Random
    signs_rand = [rng.choice((-1, 1)) for _ in range(n)]
    
    # Strategy 4: Another Random
    rng2 = random.Random(123)
    signs_rand2 = [rng2.choice((-1, 1)) for _ in range(n)]
    
    candidates = [signs_tm, signs_alt, signs_rand, signs_rand2]
    best_signs = candidates[0]
    best_disc = float('inf')
    
    for c in candidates:
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
    
    current_signs = best_signs[:]
    current_disc = best_disc
    
    # Simulated Annealing Parameters
    T = 5.0  # Initial temperature
    cooling_rate = 0.9995
    min_T = 0.1
    
    # Track stalls to trigger restarts
    stall_counter = 0
    max_stalls = 500
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.randrange(n)
        old_val = current_signs[i]
        current_signs[i] = -old_val
        
        new_disc = _max_ap_discrepancy(current_signs)
        delta = new_disc - current_disc
        
        # Acceptance probability
        if delta < 0:
            # Improvement, always accept
            current_disc = new_disc
            current_signs[i] = -current_signs[i] # Wait, we already flipped it. 
            # Actually, current_signs[i] is the new value. 
            # If we accept, we keep it.
            # If we reject, we revert.
            
            # Let's restructure the flip logic for clarity
            # Flip
            current_signs[i] = -old_val
            new_disc = _max_ap_discrepancy(current_signs)
            delta = new_disc - current_disc
            
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                # Accept
                current_disc = new_disc
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = current_signs[:]
                    stall_counter = 0
                else:
                    stall_counter += 1
            else:
                # Reject, revert
                current_signs[i] = old_val
                stall_counter += 1
        else:
            # This branch is redundant if we restructure, but keeping logic clear
            pass
            
        # Cooling
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # Restart mechanism: If stuck for a long time, reset to best and reheat
        if stall_counter > max_stalls:
            current_signs = best_signs[:]
            current_disc = best_disc
            T = 5.0
            stall_counter = 0
            
    return best_signs
# EVOLVE-BLOCK-END
