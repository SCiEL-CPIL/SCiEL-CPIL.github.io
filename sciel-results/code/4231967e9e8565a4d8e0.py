"""Rewritten program: local search minimizing max |sum| over all APs in [1,256]."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _max_abs_ap_sum_numpy(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in indices 0..n-1 using NumPy for speed.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons
    
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                m = np.max(np.abs(csum))
                if m > max_disc:
                    max_disc = m
    return max_disc

def _local_search_fast(signs, max_time, rng):
    """
    Fast local search using the numpy evaluator.
    """
    n = len(signs)
    signs = signs.copy()
    current_max = _max_abs_ap_sum_numpy(signs)
    best_max = current_max
    best_signs = signs.copy()
    
    start_time = time.time()
    deadline = start_time + max_time
    
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline:
        improved = False
        positions = list(range(n))
        rng.shuffle(positions)
        
        for pos in positions:
            if time.time() > deadline:
                break
            
            signs[pos] = -signs[pos]
            new_max = _max_abs_ap_sum_numpy(signs)
            
            if new_max < current_max:
                current_max = new_max
                improved = True
                no_improve_count = 0
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                break
            else:
                signs[pos] = -signs[pos]
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                signs = best_signs.copy()
                num_flips = rng.randint(1, 5)
                for _ in range(num_flips):
                    p = rng.randint(0, n - 1)
                    signs[p] = -signs[p]
                current_max = _max_abs_ap_sum_numpy(signs)
                no_improve_count = 0
                
    return best_signs, best_max

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    start_time = time.time()
    total_budget = 55  # 60s limit, leave 5s margin
    
    rng = random.Random(42)
    
    signs = np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int8)
    
    best_signs = signs.copy()
    best_max = _max_abs_ap_sum_numpy(signs)
    
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5:
        result_signs, result_max = _local_search_fast(signs, remaining, rng)
        if result_max < best_max:
            best_max = result_max
            best_signs = result_signs
            
    remaining2 = total_budget - (time.time() - start_time)
    if remaining2 > 10:
        signs2 = np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int8)
        result_signs2, result_max2 = _local_search_fast(signs2, remaining2, rng)
        if result_max2 < best_max:
            best_max = result_max2
            best_signs = result_signs2

    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
