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
    
    # 3. Create list of (sigma[n], n) pairs and sort by sigma, then by n
    # This avoids the overhead of defaultdict and allows efficient two-pointer search
    pairs = [(sigma[n], n) for n in range(1, MAX_N + 1)]
    pairs.sort()
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # 4. Two-pointer search for best pair
    # For each group of equal sigma, find pairs (a, b) that minimize |2a - 3b| / (2b)
    # We can use a two-pointer technique within each group
    
    best_num = 10**18
    best_den = 1
    best_a = 11
    best_b = 6
    
    i = 0
    n_pairs = len(pairs)
    while i < n_pairs:
        if time.time() > deadline - 2:
            break
            
        s_val = pairs[i][0]
        j = i
        while j < n_pairs and pairs[j][0] == s_val:
            j += 1
        
        # Group is pairs[i:j]
        group_size = j - i
        if group_size >= 2:
            # Extract the n values from this group
            # They are already sorted in increasing order
            # Use two pointers to find best pair
            # For each b, we want a ≈ 1.5*b
            # Since the group is sorted, we can use binary search or two pointers
            
            # Two-pointer approach:
            # For each index b_idx in the group, find the a that minimizes |2a - 3b|
            # Since both a and b come from the same sorted group, we can use a sliding window
            
            # Better approach: for each b, use bisect to find closest a to 1.5*b
            # But we need to be careful about time
            
            # Let's extract the n values
            group_n = [pairs[k][1] for k in range(i, j)]
            
            # For each b in group_n, find best a
            # We can iterate through group_n as potential b values
            for b_idx in range(len(group_n)):
                b_val = group_n[b_idx]
                
                # Pruning: if 1/(2*b_val) >= best_num/best_den, skip
                if best_den >= 2 * b_val * best_num:
                    continue
                    
                # Find a in group_n closest to 1.5 * b_val
                target = 1.5 * b_val
                pos = bisect.bisect_left(group_n, target)
                
                candidates = []
                if pos < len(group_n):
                    candidates.append(group_n[pos])
                if pos > 0:
                    candidates.append(group_n[pos-1])
                
                for a_val in candidates:
                    if a_val == b_val:
                        continue
                    num = abs(2 * a_val - 3 * b_val)
                    if num == 0:
                        continue
                    den = 2 * b_val
                    
                    if num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a = a_val
                        best_b = b_val
                        
        i = j
        
    if best_num == 10**18:
        return [11, 6]
        
    return [int(best_a), int(best_b)]
