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
            m = (n - 1 - r) // d + 1
            if m <= 0:
                continue
            
            # Extract subsequence
            indices = np.arange(r, n, d)
            sub = signs_arr[indices]
            
            # Compute cumulative sum
            cum = np.concatenate(([0], np.cumsum(sub)))
            
            # Find max subarray sum and min subarray sum using running min/max
            running_min = 0
            running_max = 0
            max_sum = 0
            min_sum = 0
            
            for i in range(1, m + 1):
                val = cum[i]
                if val < running_min:
                    running_min = val
                if val > running_max:
                    running_max = val
                diff_max = val - running_min
                diff_min = val - running_max
                if diff_max > max_sum:
                    max_sum = diff_max
                if diff_min < min_sum:
                    min_sum = diff_min
            
            current_max = max(abs(max_sum), abs(min_sum))
            if current_max > max_abs:
                max_abs = current_max
    
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    rng = random.Random(42)
    
    # Initialize with multiple strategies
    candidates = []
    
    # Strategy 1: Thue-Morse
    signs_tm = [0] * n
    for i in range(n):
        t = 0
        x = i
        while x:
            t ^= (x & 1)
            x >>= 1
        signs_tm[i] = 1 if t == 0 else -1
    candidates.append(signs_tm)
    
    # Strategy 2: Alternating
    signs_alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(signs_alt)
    
    # Strategy 3: Random with seed 42
    signs_rand = [rng.choice((-1, 1)) for _ in range(n)]
    candidates.append(signs_rand)
    
    # Strategy 4: Random with seed 123
    rng2 = random.Random(123)
    signs_rand2 = [rng2.choice((-1, 1)) for _ in range(n)]
    candidates.append(signs_rand2)
    
    # Strategy 5: Another random seed
    rng3 = random.Random(777)
    signs_rand3 = [rng3.choice((-1, 1)) for _ in range(n)]
    candidates.append(signs_rand3)
    
    # Evaluate all candidates and pick the best
    best_signs = candidates[0]
    best_disc = float('inf')
    
    for c in candidates:
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
    
    signs = best_signs[:]
    
    # Hill climbing with multiple passes
    improved = True
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 3  # Allow some exploration
    
    while improved and time.time() < deadline and iter_count < 20000:
        improved = False
        iter_count += 1
        
        order = list(range(n))
        rng.shuffle(order)
        
        for i in order:
            if time.time() > deadline:
                break
            
            old_val = signs[i]
            signs[i] = -old_val
            new_disc = _max_ap_discrepancy(signs)
            
            if new_disc < best_disc:
                best_disc = new_disc
                improved = True
                no_improve_count = 0
            elif new_disc == best_disc and rng.random() < 0.2:
                # Accept equal moves with some probability to escape plateaus
                improved = True
                no_improve_count = 0
            else:
                signs[i] = old_val
                no_improve_count += 1
        
        if no_improve_count >= max_no_improve * n:
            # Try a random restart from current best
            if time.time() < deadline:
                signs = best_signs[:]
                # Perturb a few random positions
                for _ in range(rng.randint(1, 3)):
                    pos = rng.randint(0, n - 1)
                    signs[pos] = -signs[pos]
                new_disc = _max_ap_discrepancy(signs)
                if new_disc <= best_disc:
                    best_signs = signs[:]
                    best_disc = new_disc
                else:
                    signs = best_signs[:]
                no_improve_count = 0
    
    # Final local search pass for polish
    if time.time() < deadline - 3:
        for i in range(n):
            if time.time() > deadline:
                break
            old_val = signs[i]
            signs[i] = -old_val
            new_disc = _max_ap_discrepancy(signs)
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs[:]
            else:
                signs[i] = old_val
    
    return best_signs
# EVOLVE-BLOCK-END
