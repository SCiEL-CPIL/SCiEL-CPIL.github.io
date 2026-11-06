import math
import time

def construct():
    start_time = time.time()
    deadline = 85  # Keep a margin for safety
    
    N = 1000000
    MAX_VAL = N + 1
    
    # 1. Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    spf = list(range(MAX_VAL + 1))
    if MAX_VAL >= 1:
        spf[0] = 0
    if MAX_VAL >= 2:
        spf[1] = 1
        
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute Powerful Part for each number using DP
    # P[i] = powerful part of i
    # If i is not divisible by p^2 where p = spf[i], then P[i] = P[i // spf[i]]
    # If i is divisible by p^2, we need to extract the full p^e factor
    P = [1] * (MAX_VAL + 1)
    
    for i in range(2, MAX_VAL + 1):
        p = spf[i]
        # Check if i is divisible by p^2
        if i % (p * p) == 0:
            # Extract all factors of p from i
            n = i
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            # The powerful part includes p^count and the powerful part of n
            P[i] = (p ** count) * P[n]
        else:
            # i has only one factor of p, so p doesn't contribute to powerful part
            P[i] = P[i // p]
        
        # Check time
        if time.time() - start_time > deadline - 10:
            break

    # If we broke early, we need to handle the rest or just return best so far.
    # Given N=10^6, the loop should finish in < 2 seconds.
    
    # 3. Find max P[n] * P[n+1]
    best_n = 10000
    best_val = -1
    
    for n in range(10000, N + 1):
        val = P[n] * P[n+1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() - start_time > deadline - 2:
            break
            
    return best_n
