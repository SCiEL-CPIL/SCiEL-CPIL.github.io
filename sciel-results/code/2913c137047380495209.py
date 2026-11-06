import time
import random
import numpy as np

def _square_nonzero_count_dense(coeffs):
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    return int(np.count_nonzero(sq))

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_coeffs = None
    best_score = float('inf')
    
    random.seed(42)
    np.random.seed(42)
    
    coeff_pool = list(range(-64, 0)) + list(range(1, 65))
    
    # Try a few different initializations and run SA
    num_restarts = 10
    iters_per_restart = 10000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random start
        current_coeffs = [random.choice(coeff_pool) for _ in range(12)]
        current_score = _square_nonzero_count_dense(current_coeffs)
        
        best_coeffs = list(current_coeffs)
        best_score = current_score
        
        T_start = 20.0
        T_end = 0.1
        
        for it in range(iters_per_restart):
            if time.time() > deadline:
                break
                
            progress = it / iters_per_restart
            T = T_start * (T_end / T_start) ** progress
            
            i = random.randint(0, 11)
            new_coeff = random.choice(coeff_pool)
            if new_coeff == current_coeffs[i]:
                continue
                
            old_coeff = current_coeffs[i]
            current_coeffs[i] = new_coeff
            
            new_score = _square_nonzero_count_dense(current_coeffs)
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_coeffs = list(current_coeffs)
            else:
                current_coeffs[i] = old_coeff
                
        # Update global best
        if best_score < best_score: # Wait, logic error in variable scope
            pass

    # Corrected logic for global best
    global_best_coeffs = None
    global_best_score = float('inf')
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_coeffs = [random.choice(coeff_pool) for _ in range(12)]
        current_score = _square_nonzero_count_dense(current_coeffs)
        
        local_best_coeffs = list(current_coeffs)
        local_best_score = current_score
        
        T_start = 20.0
        T_end = 0.1
        
        for it in range(iters_per_restart):
            if time.time() > deadline:
                break
                
            progress = it / iters_per_restart
            T = T_start * (T_end / T_start) ** progress
            
            i = random.randint(0, 11)
            new_coeff = random.choice(coeff_pool)
            if new_coeff == current_coeffs[i]:
                continue
                
            old_coeff = current_coeffs[i]
            current_coeffs[i] = new_coeff
            
            new_score = _square_nonzero_count_dense(current_coeffs)
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                if new_score < local_best_score:
                    local_best_score = new_score
                    local_best_coeffs = list(current_coeffs)
            else:
                current_coeffs[i] = old_coeff
                
        if local_best_score < global_best_score:
            global_best_score = local_best_score
            global_best_coeffs = local_best_coeffs
            
    if global_best_coeffs is None:
        return [1]*12
        
    return global_best_coeffs
