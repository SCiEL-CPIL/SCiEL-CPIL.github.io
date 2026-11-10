import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    min_n = 1000
    window_size = 24
    
    max_val = max_n + window_size
    
    small_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk <= max_val:
            small_part[pk::pk] *= p
            pk *= p
    
    small_part_list = small_part.tolist()
    
    counts = {}
    distinct_count = 0
    
    for i in range(1, window_size + 1):
        val = small_part_list[min_n + i]
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
            distinct_count += 1
            
    best_n = min_n
    best_score = distinct_count
    
    for n in range(min_n, max_n + 1):
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            
        if n < max_n:
            out_val = small_part_list[n + 1]
            in_val = small_part_list[n + window_size + 1]
            
            counts[out_val] -= 1
            if counts[out_val] == 0:
                distinct_count -= 1
                del counts[out_val]
                
            if in_val in counts:
                counts[in_val] += 1
            else:
                counts[in_val] = 1
                distinct_count += 1

        if time.time() > deadline:
            break

    return best_n
