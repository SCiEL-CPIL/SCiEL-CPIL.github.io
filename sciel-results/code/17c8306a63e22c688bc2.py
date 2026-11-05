import time
import math

def construct():
    start_time = time.time()
    # The evaluation limit is 90s. We use a generous deadline to be safe,
    # but the algorithm is very fast for N=123457.
    deadline = start_time + 80.0
    
    target = 123457
    
    # Compute largest prime factor P[n] for all n up to target
    # P[n] will store the largest prime factor of n.
    # For n=1, P[1]=1.
    P = [0] * (target + 1)
    P[1] = 1
    
    # Initialize even numbers with 2
    for j in range(2, target + 1, 2):
        P[j] = 2
        
    # Sieve for odd primes
    # We iterate p from 3 to target.
    # If P[p] is 0, p is prime.
    # We update multiples of p. Since we iterate p in increasing order,
    # the last update for any composite number n will be by its largest prime factor.
    for p in range(3, target + 1, 2):
        if P[p] == 0:  # p is prime
            for j in range(p, target + 1, p):
                P[j] = p
                
    # Now find the pair (a, b) with a+b=target minimizing max(P[a], P[b])
    best_a = 1
    best_b = target - 1
    min_max_p = max(P[1], P[target - 1])
    
    # We only need to check a from 1 to target // 2
    # Because the pair (a, b) is the same as (b, a)
    for a in range(1, target // 2 + 1):
        b = target - a
        pa = P[a]
        pb = P[b]
        current_max = pa if pa > pb else pb
        
        if current_max < min_max_p:
            min_max_p = current_max
            best_a = a
            best_b = b
            
            # If we hit the theoretical minimum for non-trivial sums, we can stop
            # But 1 is not possible for both. 2 is not possible.
            # So we just run the full scan. It's very fast.
            
        if time.time() > deadline:
            break
            
    return [best_a, best_b]
