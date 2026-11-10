import math
import time

def construct():
    N = 123457
    
    # Sieve to find largest prime factor for all numbers up to N
    P = [1] * (N + 1)
    
    # Find primes and update P
    # We can do this in one pass if we iterate i from 2 to N
    # But standard sieve for primes first is easier to reason about
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    is_prime[j] = False
                    
    # Update P array with largest prime factor
    for p in primes:
        for m in range(p, N + 1, p):
            P[m] = p
            
    best_val = float('inf')
    best_pair = [1, N - 1]
    
    for a in range(1, N // 2 + 1):
        b = N - a
        val = max(P[a], P[b])
        if val < best_val:
            best_val = val
            best_pair = [a, b]
            if best_val == 3:
                break

    return best_pair
