import time
import numpy as np
import random

_N = 256
_INDICES_CACHE = None

def _precompute_indices(n):
    data = {}
    for s in range(1, n + 1):
        max_len = (n + s - 1) // s
        rows = np.arange(s)[:, np.newaxis]
        cols = np.arange(max_len)[np.newaxis, :]
        linear_indices = rows + cols * s
        mask = linear_indices < n
        safe_indices = np.minimum(linear_indices, n - 1)
        data[s] = (safe_indices, mask)
    return data

def _get_indices():
    global _INDICES_CACHE
    if _INDICES_CACHE is None:
        _INDICES_CACHE = _precompute_indices(_N)
    return _INDICES_CACHE

def _fast_max_discrepancy(signs):
    n = len(signs)
    arr = np.asarray(signs, dtype=np.int64)
    max_disc = 1
    indices_cache = _get_indices()
    
    for s in range(1, n + 1):
        safe_indices, mask = indices_cache[s]
        M = arr[safe_indices].copy()
        M[~mask] = 0
        prefix_sums = np.cumsum(M, axis=1)
        current_max = np.max(np.abs(prefix_sums))
        if current_max > max_disc:
            max_disc = int(current_max)
            
    return max_disc

def _find_worst_ap(signs):
    n = len(signs)
    arr = np.asarray(signs, dtype=np.int64)
    max_disc = 1
    best_start = 0
    best_step = 1
    best_len = 1
    indices_cache = _get_indices()
    
    for s in range(1, n + 1):
        safe_indices, mask = indices_cache[s]
        M = arr[safe_indices].copy()
        M[~mask] = 0
        prefix_sums = np.cumsum(M, axis=1)
        abs_sums = np.abs(prefix_sums)
        
        # Find the max in this step's matrix
        max_val = np.max(abs_sums)
        if max_val > max_disc:
            max_disc = int(max_val)
            # Find which row and column
            flat_idx = np.argmax(abs_sums)
            row_idx = flat_idx // abs_sums.shape[1]
            col_idx = flat_idx % abs_sums.shape[1]
            best_step = s
            best_start = row_idx
            best_len = col_idx + 1
            
    return max_disc, best_start, best_step, best_len

def _thue_morse(n):
    signs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        c = bin(i).count('1')
        signs[i] = 1 if c % 2 == 0 else -1
    return signs

def _legendre(n, p=257):
    signs = np.zeros(n, dtype=np.int64)
    for i in range(1, n+1):
        val = pow(i, (p-1)//2, p)
        if val == 1:
            signs[i-1] = 1
        else:
            signs[i-1] = -1
    return signs

def _balanced_init(n, seed=42):
    """Initialize with a sequence that tries to balance AP sums using a greedy approach."""
    rng = random.Random(seed)
    signs = np.zeros(n, dtype=np.int64)
    
    # Start with Thue-Morse as base, then do a few passes of worst-AP flipping
    signs = _thue_morse(n)
    
    for _ in range(200):
        disc, start, step, length = _find_worst_ap(signs)
        if disc <= 2:
            break
        ap_indices = list(range(start, start + length * step, step))
        current_sum = sum(signs[i] for i in ap_indices)
        target_sign = 1 if current_sum > 0 else -1
        candidates = [i for i in ap_indices if signs[i] == target_sign]
        if candidates:
            idx = rng.choice(candidates)
            signs[idx] = -signs[idx]
    
    return signs

def _local_search(signs, time_limit=55.0, seed=42):
    n = len(signs)
    signs = np.array(signs, dtype=np.int64)
    current_disc = _fast_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    rng = random.Random(seed)
    
    no_improve_count = 0
    max_no_improve = 30
    
    T = 5.0
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        # Find worst AP
        disc, start, step, length = _find_worst_ap(signs)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = signs.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, try targeted flip in worst AP
        if no_improve_count > max_no_improve:
            ap_indices = list(range(start, start + length * step, step))
            current_sum = sum(signs[i] for i in ap_indices)
            target_sign = 1 if current_sum > 0 else -1
            candidates = [i for i in ap_indices if signs[i] == target_sign]
            
            if candidates:
                idx = rng.choice(candidates)
            else:
                idx = rng.randint(0, n - 1)
            
            old_sign = signs[idx]
            signs[idx] = -old_sign
            new_disc = _fast_max_discrepancy(signs)
            
            if new_disc <= current_disc or (T > 0.01 and rng.random() < np.exp(-(new_disc - current_disc) / T)):
                current_disc = new_disc
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs.copy()
                    no_improve_count = 0
            else:
                signs[idx] = old_sign
                no_improve_count += 1
                
            T *= alpha
        else:
            # Normal SA move: random flip
            idx = rng.randint(0, n - 1)
            old_sign = signs[idx]
            signs[idx] = -old_sign
            
            new_disc = _fast_max_discrepancy(signs)
            delta = new_disc - current_disc
            
            if delta < 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
                current_disc = new_disc
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs.copy()
            else:
                signs[idx] = old_sign
                
            T *= alpha
            
    return best_signs.tolist(), best_disc

def construct():
    n = _N
    start_time = time.time()
    deadline = 58.0
    
    best_signs = None
    best_disc = float('inf')
    
    inits = [
        _thue_morse(n),
        _legendre(n),
        _balanced_init(n, seed=42),
        _balanced_init(n, seed=123),
        np.random.RandomState(1).choice([-1, 1], n),
        np.random.RandomState(7).choice([-1, 1], n),
    ]
    
    # Evaluate initial discrepancies to sort them
    init_discs = []
    for init_signs in inits:
        d = _fast_max_discrepancy(init_signs)
        init_discs.append(d)
        if d < best_disc:
            best_disc = d
            best_signs = init_signs.tolist()
    
    # Sort initializations by discrepancy (best first)
    sorted_inits = sorted(zip(inits, init_discs), key=lambda x: x[1])
    
    time_per_run = (deadline - (time.time() - start_time)) / len(sorted_inits)
    
    for i, (init_signs, init_disc) in enumerate(sorted_inits):
        if time.time() - start_time > deadline:
            break
            
        current_time_limit = time_per_run
        if time.time() - start_time + current_time_limit > deadline:
             current_time_limit = deadline - (time.time() - start_time)
             
        if current_time_limit <= 0:
            break

        res_signs, res_disc = _local_search(init_signs, time_limit=current_time_limit, seed=42 + i)
        
        if res_disc < best_disc:
            best_disc = res_disc
            best_signs = res_signs
            
    if best_signs is None:
        return [1] * n
        
    return best_signs
