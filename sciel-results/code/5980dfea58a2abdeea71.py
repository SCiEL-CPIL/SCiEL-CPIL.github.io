import numpy as np
import time

def _compute_max_abs_ap_sum(signs):
    n = len(signs)
    max_abs = 0
    
    for s in range(1, n):
        max_len = (n - 1) // s + 1
        if max_len <= max_abs:
            break
        
        for r in range(s):
            subseq = signs[r::s].astype(np.int32)
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_abs:
                max_abs = int(current_max)
                
    return max_abs

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s margin
    
    rng = np.random.RandomState(42)
    
    best_signs = None
    best_score = float('inf')
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            signs = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        elif restart == 1:
            signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        else:
            signs = rng.choice([-1, 1], size=n).astype(np.int8)
        
        cur_score = _compute_max_abs_ap_sum(signs)
        
        if cur_score < best_score:
            best_score = cur_score
            best_signs = signs.copy()
        
        improved = True
        passes = 0
        max_passes = 20
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            idx_order = rng.permutation(n)
            
            for i in idx_order:
                if time.time() > deadline - 2:
                    break
                
                old_val = signs[i]
                signs[i] = -old_val
                new_score = _compute_max_abs_ap_sum(signs)
                
                if new_score < cur_score:
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_signs = signs.copy()
                    improved = True
                else:
                    signs[i] = old_val
            
            if time.time() > deadline - 2:
                break
                
    if best_signs is None:
        best_signs = np.ones(n, dtype=np.int8)
        
    return [int(x) for x in best_signs]
