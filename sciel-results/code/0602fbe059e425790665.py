# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # 1. Generate primes <= 10000
    MAX_VAL = 10000
    is_prime = np.ones(MAX_VAL + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    N = len(primes)
    P = np.array(primes, dtype=np.int32)
    
    MAX_GAP = 10000
    
    # 2. DP Arrays
    pref_max = np.zeros((N, MAX_GAP + 1), dtype=np.int32)
    arg_gap = np.zeros((N, MAX_GAP + 1), dtype=np.int32)
    parent_idx = np.full((N, MAX_GAP + 1), -1, dtype=np.int32)
    parent_gap = np.zeros((N, MAX_GAP + 1), dtype=np.int32)
    
    # Reusable temp arrays
    temp_arr = np.zeros(MAX_GAP + 1, dtype=np.int32)
    temp_p_idx = np.full(MAX_GAP + 1, -1, dtype=np.int32)
    temp_p_gap = np.zeros(MAX_GAP + 1, dtype=np.int32)
    
    for i in range(N):
        if time.time() > deadline:
            break
            
        # Reset temp arrays
        temp_arr[:] = 0
        temp_p_idx[:] = -1
        temp_p_gap[:] = 0
        
        # Base case: length 1
        temp_arr[0] = 1
        
        # Iterate predecessors
        for j in range(i):
            if time.time() > deadline:
                break
            g = P[i] - P[j]
            if g > MAX_GAP: continue
            
            best_len_j = pref_max[j][g]
            prev_g = arg_gap[j][g]
            new_len = best_len_j + 1
            
            if new_len > temp_arr[g]:
                temp_arr[g] = new_len
                temp_p_idx[g] = j
                temp_p_gap[g] = prev_g
                
        # Build prefix max
        curr_max = 0
        curr_arg = 0
        curr_p_idx = -1
        curr_p_gap = 0
        
        for g in range(MAX_GAP + 1):
            if temp_arr[g] > curr_max:
                curr_max = temp_arr[g]
                curr_arg = g
                curr_p_idx = temp_p_idx[g]
                curr_p_gap = temp_p_gap[g]
            
            pref_max[i][g] = curr_max
            arg_gap[i][g] = curr_arg
            parent_idx[i][g] = curr_p_idx
            parent_gap[i][g] = curr_p_gap
            
    # Find best
    max_lengths = pref_max[:, MAX_GAP]
    best_i = np.argmax(max_lengths)
    best_len = max_lengths[best_i]
    best_g = arg_gap[best_i][MAX_GAP]
    
    # Backtrack
    path = []
    i = best_i
    g = best_g
    length = best_len
    
    while length > 1:
        path.append(P[i])
        j = parent_idx[i][g]
        prev_g = parent_gap[i][g]
        i = j
        g = prev_g
        length -= 1
        if time.time() > deadline:
            break
            
    path.append(P[i])
    path.reverse()
    
    return path
# EVOLVE-BLOCK-END
