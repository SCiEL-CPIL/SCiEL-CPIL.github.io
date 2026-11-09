import numpy as np
import time

N = 100

def _calc_max_overlap(h):
    g = 1 - h
    corr = np.correlate(h, g, mode='valid')
    return np.max(corr)

def _run_sa(h_init, time_limit):
    start = time.time()
    h = h_init.copy()
    current_score = _calc_max_overlap(h)
    best_score = current_score
    best_h = h.copy()
    
    T = 5.0
    T_min = 0.5
    cooling = 0.9995
    
    while time.time() - start < time_limit:
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        h[i], h[j] = h[j], h[i]
        
        new_score = _calc_max_overlap(h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            
        if new_score < current_score:
            current_score = new_score
        else:
            delta = new_score - current_score
            if np.random.rand() < np.exp(-delta / T):
                current_score = new_score
            else:
                h[i], h[j] = h[j], h[i]
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_h, best_score

def construct():
    global_start = time.time()
    deadline = 115.0 # 5s buffer
    
    best_h_global = None
    best_score_global = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline:
        # Create initial random sequence
        h_init = np.zeros(N, dtype=int)
        indices = np.random.permutation(N)
        h_init[indices[:N//2]] = 1
        
        # Allocate remaining time to this SA run, min 1s
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        run_time = min(remaining, 10.0) # Cap each run at 10s to allow restarts
        
        h_res, score_res = _run_sa(h_init, run_time)
        
        if score_res < best_score_global:
            best_score_global = score_res
            best_h_global = h_res
            
        if time.time() > deadline - 2.0:
            break

    if best_h_global is None:
        # Fallback to constant
        return np.full(N, 0.5)
        
    return best_h_global.astype(float)

# EVOLVE-BLOCK-END
