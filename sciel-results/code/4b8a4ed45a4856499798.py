import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    max_val = max_n + 24
    
    small_factor = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= max_val:
            for j in range(power, max_val + 1, power):
                small_factor[j] *= p
            power *= p
            
    sf = small_factor[1001:max_n + 25]
    
    best_score = 25
    best_n = 1000
    
    counts = {}
    current_distinct = 0
    
    for i in range(24):
        v = int(sf[i])
        if v not in counts:
            counts[v] = 1
            current_distinct += 1
        else:
            counts[v] += 1
            
    if current_distinct < best_score:
        best_score = current_distinct
        best_n = 1000
        if best_score == 1:
            return best_n
            
    for n in range(1, max_n - 1000):
        if n % 100000 == 0:
            if time.time() > deadline:
                break
                
        val_out = int(sf[n])
        c = counts[val_out] - 1
        if c == 0:
            del counts[val_out]
            current_distinct -= 1
        else:
            counts[val_out] = c
            
        val_in = int(sf[n + 24])
        if val_in not in counts:
            counts[val_in] = 1
            current_distinct += 1
        else:
            counts[val_in] += 1
            
        if current_distinct < best_score:
            best_score = current_distinct
            best_n = 1000 + n
            if best_score == 1:
                break
                
    return best_n
