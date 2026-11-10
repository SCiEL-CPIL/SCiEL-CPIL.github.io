import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    # SPF[n] will store the smallest prime factor of n
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def largest_prime_factor(n):
        if n < 2:
            return 1
        max_p = 1
        while n > 1:
            p = spf[n]
            if p > max_p:
                max_p = p
            while n % p == 0:
                n //= p
        return max_p

    best_cost = float('inf')
    best_pair = (1, N - 1)
    
    # Iterate a from 1 to N//2
    # We can stop early if we find cost 1 (impossible for N>2 as P(1)=1, P(N-1)>=2)
    # Min possible cost is 2 (if N is even, a=2, b=N-2? No, P(2)=2, P(N-2) might be large)
    # Actually, if N is odd, a+b=N.
    
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        
        b = N - a
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        cost = pa if pa > pb else pb
        
        if cost < best_cost:
            best_cost = cost
            best_pair = (a, b)
            
            # If we find cost 2, it's the theoretical minimum for N>3?
            # P(a)=2 implies a is power of 2.
            # P(b)=2 implies b is power of 2.
            # Sum of two powers of 2 is N.
            # If N is not a sum of two powers of 2, cost > 2.
            # If we find cost 2, we can break.
            if best_cost == 2:
                break

    return [best_pair[0], best_pair[1]]
