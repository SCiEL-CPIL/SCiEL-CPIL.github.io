import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 75.0  # Safety margin before 90s kill

    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p : limit+1 : p] = False
    
    primes = np.where(is_prime)[0]
    N = len(primes)
    
    if N == 0:
        return []
    
    # prev_best[i, g] = max length of valid sequence ending at primes[i] with last gap <= g
    prev_best = np.zeros((N, limit + 1), dtype=np.uint8)
    
    # For reconstruction:
    # parent_idx[i, g] = index j of the previous prime
    # parent_gap[i, g] = the actual gap used (primes[i] - primes[j])
    parent_idx = np.full((N, limit + 1), -1, dtype=np.int32)
    parent_gap = np.zeros((N, limit + 1), dtype=np.uint16)
    
    # Base case: i=0
    prev_best[0, :] = 1
    
    global_max_len = 1
    best_end_i = 0
    best_end_g = 0
    
    for i in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        
        pi = int(primes[i])
        
        # L[d] = max length ending at i with last gap exactly d
        L = np.zeros(limit + 1, dtype=np.uint8)
        best_j_for_gap = np.full(limit + 1, -1, dtype=np.int32)
        
        for j in range(i):
            gap = pi - int(primes[j])
            if gap <= 0 or gap > limit:
                continue
            val = prev_best[j, gap]
            if val > 0:
                new_len = val + 1
                if new_len > L[gap]:
                    L[gap] = new_len
                    best_j_for_gap[gap] = j
        
        # Build prefix max
        cummax = np.cummax(L)
        prev_best[i, :] = cummax
        
        # For reconstruction: find for each g, the gap that achieved the running max
        # and the corresponding j
        current_max = 0
        current_arg = 0
        current_p_idx = -1
        current_p_gap = 0
        
        for g in range(1, limit + 1):
            if L[g] > current_max:
                current_max = L[g]
                current_arg = g
                current_p_idx = best_j_for_gap[g]
                current_p_gap = g
            
            parent_idx[i, g] = current_p_idx
            parent_gap[i, g] = current_p_gap
        
        # Update global best
        if cummax[-1] > global_max_len:
            global_max_len = cummax[-1]
            best_end_i = i
            # Find a gap g where the max is achieved
            max_positions = np.where(cummax == cummax[-1])[0]
            best_end_g = int(max_positions[-1]) if len(max_positions) > 0 else 0

    # Reconstruct
    if global_max_len <= 1:
        return [int(primes[0])]
    
    seq = []
    cur_i = best_end_i
    cur_g = best_end_g
    
    while cur_i >= 0:
        seq.append(int(primes[cur_i]))
        if cur_i == 0:
            break
        next_i = int(parent_idx[cur_i, cur_g])
        next_g = int(parent_gap[cur_i, cur_g])
        cur_i = next_i
        cur_g = next_g
    
    seq.reverse()
    return seq
