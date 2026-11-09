import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need smooth parts for numbers up to 1000000 + 24 = 1000024
    max_val = 1000024
    sp = [1] * (max_val + 1)
    
    # Sieve: for each small prime p, multiply sp[m] by p^exp for each multiple m of p
    for p in small_primes:
        for m in range(p, max_val + 1, p):
            temp = m
            exp = 0
            while temp % p == 0:
                exp += 1
                temp //= p
            if exp > 0:
                sp[m] *= (p ** exp)
    
    # Map smooth parts to compact integer IDs for faster counting
    unique_sp = list(set(sp[1:]))
    sp_id_map = {v: i for i, v in enumerate(unique_sp)}
    num_unique = len(unique_sp)
    
    # Convert sp to ID array
    sp_ids = [0] * (max_val + 1)
    for i in range(1, max_val + 1):
        sp_ids[i] = sp_id_map[sp[i]]
    
    best_n = 1000
    best_count = float('inf')
    
    # Sliding window using frequency array
    freq = [0] * num_unique
    distinct = 0
    
    # Initialize for n=1000: window [1001, 1024]
    for i in range(1001, 1025):
        idx = sp_ids[i]
        if freq[idx] == 0:
            distinct += 1
        freq[idx] += 1
    
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    if best_count == 1:
        return best_n
    
    # Slide window: for new_n from 1001 to 1000000
    for new_n in range(1001, 1000001):
        if time.time() > deadline:
            break
        
        # Remove leftmost: sp[new_n]
        old_idx = sp_ids[new_n]
        freq[old_idx] -= 1
        if freq[old_idx] == 0:
            distinct -= 1
        
        # Add new rightmost: sp[new_n + 24]
        new_idx = sp_ids[new_n + 24]
        if freq[new_idx] == 0:
            distinct += 1
        freq[new_idx] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = new_n
            if best_count == 1:
                break
    
    return best_n
