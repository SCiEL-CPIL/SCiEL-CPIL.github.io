import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
        
    # Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def compute_phi_range(L, W):
        vals = np.arange(L, L+W, dtype=np.int64)
        phis = vals.copy()
        
        max_val = L + W - 1
        sqrt_max = int(math.isqrt(max_val))
        
        for p in primes:
            if p > sqrt_max:
                break
            
            start_val = ((L + p - 1) // p) * p
            if start_val > max_val:
                continue
            start_idx = start_val - L
            indices = np.arange(start_idx, W, p)
            
            if len(indices) == 0:
                continue
                
            phis[indices] -= phis[indices] // p
            
            v = vals[indices]
            while True:
                mask_v = v % p == 0
                if not np.any(mask_v):
                    break
                v[mask_v] //= p
            vals[indices] = v
            
        mask_rem = vals > 1
        if np.any(mask_rem):
            idx_rem = np.where(mask_rem)[0]
            phis[idx_rem] -= phis[idx_rem] // vals[idx_rem]
            
        return phis

    def find_longest_run(phis):
        if len(phis) == 0:
            return 0, 0
        last_seen = {}
        start = 0
        max_len = 0
        best_start = 0
        for i, val in enumerate(phis):
            # Convert numpy int to python int for dict key
            v = int(val)
            if v in last_seen and last_seen[v] >= start:
                start = last_seen[v] + 1
            last_seen[v] = i
            curr_len = i - start + 1
            if curr_len > max_len:
                max_len = curr_len
                best_start = start
        return max_len, best_start

    best_k = 1
    best_s = 1
    
    # Small range search
    W_small = 20000
    phis_small = compute_phi_range(1, W_small)
    k_small, s_off_small = find_longest_run(phis_small)
    if k_small > best_k:
        best_k = k_small
        best_s = 1 + s_off_small
        
    # Large range search
    W_large = 10000
    MAX_N = 10**12
    rng = random.Random(42)
    
    while time_remaining() > 5.0:
        L = rng.randint(1, MAX_N - W_large + 1)
        phis = compute_phi_range(L, W_large)
        k, s_off = find_longest_run(phis)
        if k > best_k:
            best_k = k
            best_s = L + s_off
            
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
