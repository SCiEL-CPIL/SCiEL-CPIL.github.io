import time
import random
import numpy as np

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute sum over any arithmetic progression
    contained in [1, n].
    signs: list or numpy array of length n, values -1 or 1.
    Returns the max absolute sum.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_sum = 0
    
    for d in range(1, n + 1):
        # For each step d, process all chains s=0..d-1
        # Vectorize over chains where possible
        for s in range(d):
            # Get indices: s, s+d, s+2d, ...
            idx = np.arange(s, n, d)
            if len(idx) == 0:
                continue
            chain = signs_arr[idx]
            # Prefix sums
            prefix = np.concatenate(([0], np.cumsum(chain)))
            current_max = int(prefix.max() - prefix.min())
            if current_max > max_sum:
                max_sum = current_max
    
    return max_sum

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 256
    rng = random.Random(42)
    
    # Initial guess: random
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_signs = signs[:]
    best_score = _max_abs_ap_sum(best_signs)
    
    # Local Search with better strategies
    current_signs = best_signs[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 200  # Restart if stuck
    
    # Pre-compute which APs each index belongs to for faster evaluation?
    # For now, use the vectorized evaluator.
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with random
            current_signs = [rng.choice((-1, 1)) for _ in range(n)]
            current_score = _max_abs_ap_sum(current_signs)
            if current_score < best_score:
                best_signs = current_signs[:]
                best_score = current_score
            no_improve_count = 0
            continue
        
        # Try flipping a random index
        idx = rng.randint(0, n - 1)
        old_val = current_signs[idx]
        current_signs[idx] = -old_val
        
        new_score = _max_abs_ap_sum(current_signs)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_signs = current_signs[:]
        else:
            # Revert
            current_signs[idx] = old_val
            no_improve_count += 1
            
    return best_signs
