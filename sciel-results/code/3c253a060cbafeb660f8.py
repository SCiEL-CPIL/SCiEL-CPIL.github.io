import time
import random
import numpy as np

def _compute_chain_max(signs, s, d):
    """Compute max absolute prefix sum for chain starting at s with step d."""
    chain = signs[s::d]
    if len(chain) == 0:
        return 0
    p = np.cumsum(chain)
    return int(np.max(np.abs(p)))

def _initialize_maxes(signs, n):
    """Initialize the maxes array for all (d, s) pairs."""
    maxes = [[0] * d for d in range(1, n + 1)]
    global_max = 0
    for d in range(1, n + 1):
        for s in range(d):
            m = _compute_chain_max(signs, s, d)
            maxes[d][s] = m
            if m > global_max:
                global_max = m
    return maxes, global_max

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 256
    rng = random.Random(42)
    
    # Generate initial candidates
    signs_rand = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    
    # Thue-Morse
    tm = []
    for i in range(n):
        c = bin(i).count('1')
        tm.append(1 if c % 2 == 0 else -1)
    signs_tm = np.array(tm, dtype=np.int8)
    
    # Alternating
    signs_alt = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    
    candidates = [signs_rand, signs_tm, signs_alt]
    
    best_signs = None
    best_score = float('inf')
    
    for cand in candidates:
        maxes, g_max = _initialize_maxes(cand, n)
        if g_max < best_score:
            best_score = g_max
            best_signs = cand.copy()
            
    current_signs = best_signs.copy()
    current_maxes, current_global_max = _initialize_maxes(current_signs, n)
    
    # Simulated Annealing
    T_start = 10.0
    T_end = 0.1
    steps = 15000  # Conservative estimate
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (step / steps)
        
        idx = rng.randint(0, n - 1)
        
        old_val = current_signs[idx]
        new_val = -old_val
        
        current_signs[idx] = new_val
        
        old_maxes_temp = []
        new_global_max = 0
        
        for d in range(1, n + 1):
            s = idx % d
            old_m = current_maxes[d][s]
            old_maxes_temp.append((d, s, old_m))
            
            new_m = _compute_chain_max(current_signs, s, d)
            current_maxes[d][s] = new_m
            
            if new_m > new_global_max:
                new_global_max = new_m
        
        delta = new_global_max - current_global_max
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_global_max = new_global_max
            if current_global_max < best_score:
                best_score = current_global_max
                best_signs = current_signs.copy()
        else:
            # Revert
            current_signs[idx] = old_val
            for d, s, old_m in old_maxes_temp:
                current_maxes[d][s] = old_m
            # current_global_max remains unchanged

    return list(best_signs)
