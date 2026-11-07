import math
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Precomputation
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    # Bitmasks for divisors
    divisors = {}
    for p in primes:
        mask = 0
        for multiple in range(p, N + 1, p):
            mask |= (1 << (multiple - 1))
        divisors[p] = mask

    weights = [i + 1 for i in range(N)]
    
    # Conflict masks
    conflict_mask = [0] * N
    for i in range(N):
        num = i + 1
        mask = 0
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= divisors[p]
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= divisors[temp]
        conflict_mask[i] = mask

    best_sum = 0
    best_indices = []
    current_indices = []
    
    def get_max_weight(mask):
        max_w = 0
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            if weights[idx] > max_w:
                max_w = weights[idx]
            m ^= lsb
        return max_w

    def calc_ub(mask, current_sum):
        ub = current_sum
        for p in primes:
            cand = mask & divisors[p]
            if cand:
                ub += get_max_weight(cand)
        return ub

    def search(mask, current_sum):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return

        if mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices.copy()
            return

        ub = calc_ub(mask, current_sum)
        if ub <= best_sum:
            return

        # Select vertex with highest weight
        max_w = -1
        chosen_idx = -1
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            if weights[idx] > max_w:
                max_w = weights[idx]
                chosen_idx = idx
            m ^= lsb
        
        if chosen_idx == -1:
            return
            
        # Branch 1: Include
        new_mask_include = mask & ~conflict_mask[chosen_idx]
        current_indices.append(chosen_idx)
        search(new_mask_include, current_sum + weights[chosen_idx])
        current_indices.pop()
        
        # Branch 2: Exclude
        new_mask_exclude = mask & ~(1 << chosen_idx)
        search(new_mask_exclude, current_sum)

    full_mask = (1 << N) - 1
    search(full_mask, 0)
    
    result = [idx + 1 for idx in best_indices]
    return [int(x) for x in result]
