import sys
import math
import time
from collections import defaultdict
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin for 90s limit
    
    MAX_N = 10_000_000
    
    # 1. Build SPF (smallest prime factor) sieve
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute sigma using multiplicative property with SPF
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    rem = [0] * (MAX_N + 1)
    rem[1] = 1
    spow = [1] * (MAX_N + 1)
    
    for n in range(2, MAX_N + 1):
        if time.time() > deadline - 15:
            # Fallback if sigma calc is too slow
            return [11, 6]
        p = spf[n]
        prev = n // p
        if prev % p == 0:
            spow[n] = spow[prev] * p
            rem[n] = rem[prev]
            sigma[n] = sigma[prev] + spow[n] * sigma[rem[n]]
        else:
            spow[n] = p
            rem[n] = prev
            sigma[n] = (1 + p) * sigma[prev]
            
    if time.time() > deadline - 10:
        return [11, 6]
    
    # 3. Group numbers by sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
        
    if time.time() > deadline - 5:
        return [11, 6]
    
    # 4. Search for best pair
    # We iterate b from MAX_N down to 1.
    # We prune if the theoretical minimum error 1/(2b) is >= current best error.
    
    best_num = 10**18
    best_den = 1
    best_a = 11
    best_b = 6
    
    # To speed up, we can pre-convert groups to lists (already done by defaultdict)
    # bisect works on lists.
    
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline - 2:
            break
            
        # Pruning check:
        # If 1 / (2*b) >= best_num / best_den
        # => best_den >= 2*b*best_num
        if best_den >= 2 * b * best_num:
            break
            
        s_val = sigma[b]
        group = sigma_groups[s_val]
        
        if len(group) < 2:
            continue
            
        # group is already sorted because we added n in increasing order
        target = 1.5 * b
        pos = bisect.bisect_left(group, target)
        
        # Check candidates around pos
        candidates = []
        if pos < len(group):
            candidates.append(group[pos])
        if pos > 0:
            candidates.append(group[pos-1])
        # Just in case, check one more step? 
        # The closest integer to 1.5b is either floor or ceil.
        # Since group contains integers, the closest member is at pos or pos-1.
        # However, if target is exactly an integer in the group, pos points to it.
        # If target is between integers, pos points to the next integer.
        # So pos and pos-1 are sufficient.
        
        for a in candidates:
            if a == b:
                continue
            num = abs(2 * a - 3 * b)
            if num == 0:
                continue
            den = 2 * b
            
            # Compare num/den with best_num/best_den
            if num * best_den < best_num * den:
                best_num = num
                best_den = den
                best_a = a
                best_b = b
                
    if best_num == 10**18:
        return [11, 6]
        
    return [int(best_a), int(best_b)]
