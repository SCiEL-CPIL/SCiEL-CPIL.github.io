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
        M = arr[safe_indices]
        M[~mask] = 0
        prefix_sums = np.cumsum(M, axis=1)
        current_max = np.max(np.abs(prefix_sums))
        if current_max > max_disc:
            max_disc = int(current_max)
            
    return max_disc

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

def _simulated_annealing(signs, time_limit=55.0):
    n = len(signs)
    signs = np.array(signs, dtype=np.int64)
    current_disc = _fast_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    rng = random.Random(42)
    
    T = 5.0
    alpha = 0.9995
    
    while time.time() - start_time < time_limit:
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
        np.random.RandomState(1).choice([-1, 1], n),
        _thue_morse(n),
        _legendre(n)
    ]
    
    for i in range(3):
        inits.append(np.random.RandomState(100 + i).choice([-1, 1], n))
        
    time_per_run = (deadline - (time.time() - start_time)) / len(inits)
    
    for init_signs in inits:
        if time.time() - start_time > deadline:
            break
            
        current_time_limit = time_per_run
        if time.time() - start_time + current_time_limit > deadline:
             current_time_limit = deadline - (time.time() - start_time)
             
        if current_time_limit <= 0:
            break

        res_signs, res_disc = _simulated_annealing(init_signs, time_limit=current_time_limit)
        
        if res_disc < best_disc:
            best_disc = res_disc
            best_signs = res_signs
            
    if best_signs is None:
        return [1] * n
        
    return best_signs
