import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0

def _max_abs_ap_sum(signs):
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int32)
    max_disc = 1
    
    for d in range(1, n):
        for r in range(d):
            seq = signs_arr[r::d]
            m = len(seq)
            if m <= 1:
                continue
            p = np.zeros(m + 1, dtype=np.int32)
            np.cumsum(seq, out=p[1:])
            
            prefix_min = np.empty(m + 1, dtype=np.int32)
            np.minimum.accumulate(p, out=prefix_min)
            prefix_min[1:] = prefix_min[:-1]
            
            prefix_max = np.empty(m + 1, dtype=np.int32)
            np.maximum.accumulate(p, out=prefix_max)
            prefix_max[1:] = prefix_max[:-1]
            
            d1 = p[1:] - prefix_min[1:]
            d2 = prefix_max[1:] - p[1:]
            best = max(int(np.max(d1)), int(np.max(d2)))
            
            if best > max_disc:
                max_disc = best
    
    return max_disc

def construct():
    start_time = time.time()
    best_signs = None
    best_val = float('inf')
    
    for restart in range(20):
        if time.time() - start_time > _DEADLINE - 5:
            break
        if best_val <= 1:
            break
        
        rng = random.Random(171 + restart)
        signs = [rng.choice((-1, 1)) for _ in range(_N)]
        val = _max_abs_ap_sum(signs)
        
        if val < best_val:
            best_val = val
            best_signs = signs[:]
        
        # Local search
        improved = True
        while improved and time.time() - start_time < _DEADLINE - 3:
            if val <= 1:
                break
            improved = False
            best_flip = -1
            best_flip_val = val
            
            for i in range(_N):
                if time.time() - start_time > _DEADLINE - 3:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                new_val = _max_abs_ap_sum(signs)
                signs[i] = old_sign
                
                if new_val < best_flip_val:
                    best_flip_val = new_val
                    best_flip = i
            
            if best_flip >= 0:
                signs[best_flip] = -signs[best_flip]
                val = best_flip_val
                improved = True
                if val < best_val:
                    best_val = val
                    best_signs = signs[:]
    
    return best_signs
