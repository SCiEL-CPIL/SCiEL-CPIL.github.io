import sys
import math
import time
from collections import defaultdict
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin for 90s limit
    
    MAX_N = 10_000_000
    
    # Build SPF (smallest prime factor) sieve
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute pow_p and rem for efficient sigma calculation
    pow_p = [1] * (MAX_N + 1)
    rem = [1] * (MAX_N + 1)
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    pow_p[1] = 1
    rem[1] = 1
    
    for n in range(2, MAX_N + 1):
        if time.time() > deadline - 10:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            pow_p[n] = pow_p[m] * p
            rem[n] = rem[m]
        else:
            pow_p[n] = p
            rem[n] = m
            
        # sigma(n) = (p^(k+1) - 1) / (p - 1) * sigma(rem)
        # pow_p[n] is p^k
        # term = (pow_p[n] * p - 1) // (p - 1)
        if p == 1:
             term = pow_p[n]
        else:
             term = (pow_p[n] * p - 1) // (p - 1)
        sigma[n] = term * sigma[rem[n]]
    
    if time.time() > deadline - 3:
        return [11, 6]
    
    # Search for minimal k = |2a - 3b|
    # We iterate k from 1 upwards
    # For each k, we check if there exists a, b <= MAX_N with 2a - 3b = k or 2a - 3b = -k
    # such that sigma(a) == sigma(b)
    
    # To speed up the inner loop, we can check time periodically
    for k in range(1, 1000000):
        if time.time() > deadline - 2:
            break
            
        # Case 1: 2a - 3b = k
        # 2a = 3b + k
        # a = (3b + k) / 2
        # b must have same parity as k for a to be integer
        # 1 <= a <= MAX_N => 2 <= 3b + k <= 2*MAX_N => (2-k)/3 <= b <= (2*MAX_N - k)/3
        b_low = max(1, (2 - k + 2) // 3)
        b_high = min(MAX_N, (2 * MAX_N - k) // 3)
        
        if b_low <= b_high:
            # Adjust b_low to match parity of k
            if (b_low - k) % 2 != 0:
                b_low += 1
                
            for b in range(b_low, b_high + 1, 2):
                a = (3 * b + k) // 2
                if a > MAX_N:
                    break
                if sigma[a] == sigma[b]:
                    return [int(a), int(b)]
                    
        # Case 2: 2a - 3b = -k => 3b - 2a = k
        # 2a = 3b - k
        # a = (3b - k) / 2
        # b must have same parity as k
        # 1 <= a <= MAX_N => 2 <= 3b - k <= 2*MAX_N => (2+k)/3 <= b <= (2*MAX_N + k)/3
        b_low = max(1, (2 + k + 2) // 3)
        b_high = min(MAX_N, (2 * MAX_N + k) // 3)
        
        if b_low <= b_high:
            if (b_low - k) % 2 != 0:
                b_low += 1
                
            for b in range(b_low, b_high + 1, 2):
                a = (3 * b - k) // 2
                if a < 1:
                    continue
                if a > MAX_N:
                    break
                if sigma[a] == sigma[b]:
                    return [int(a), int(b)]
                    
    return [11, 6]
