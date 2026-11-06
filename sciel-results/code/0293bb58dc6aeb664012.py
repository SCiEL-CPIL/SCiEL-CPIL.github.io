import time
import random
import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    best_set = set()
    best_size = 0
    
    max_restarts = 200 # Will be cut by deadline
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        current_set = set(range(N + 1))
        
        while True:
            if time.time() > deadline:
                break
                
            k = len(current_set)
            if k == 0:
                break
            
            arr = np.array(list(current_set), dtype=np.int32)
            
            bad_primes_info = []
            
            for p in PRIMES:
                if p > k:
                    break
                
                residues = arr % p
                counts = np.bincount(residues, minlength=p)
                
                min_c = np.min(counts)
                if min_c > 0:
                    min_indices = np.where(counts == min_c)[0]
                    r_to_remove = min_indices[random.randint(0, len(min_indices)-1)]
                    bad_primes_info.append((p, int(r_to_remove), int(min_c)))
            
            if not bad_primes_info:
                break
            
            bad_primes_info.sort(key=lambda x: x[2])
            p_star, r_star, c_star = bad_primes_info[0]
            
            mask = (arr % p_star) != r_star
            new_arr = arr[mask]
            current_set = set(new_arr.tolist())
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return sorted(best_set)
