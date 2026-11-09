import time
import math

def construct():
    N = 123457
    
    # Step 1: Sieve of Eratosthenes to find smallest prime factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Compute Largest Prime Factor (LPF) for each number
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        # Factorize i using spf
        x = i
        max_p = 1
        while x > 1:
            p = spf[x]
            if p > max_p:
                max_p = p
            while x % p == 0:
                x //= p
        lpf[i] = max_p
    
    # Step 3: Find the pair (a, b) minimizing max(lpf[a], lpf[b])
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    # We can optimize by iterating a from 1 to N//2 and checking symmetry, 
    # but N is small enough that full iteration is fine.
    for a in range(1, N):
        b = N - a
        val = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            # Early exit if we find 2? The minimum possible LPF for a number > 1 is 2.
            # If both are powers of 2, max is 2. 
            # But 123457 is odd, so a and b have different parity. One is even, one is odd.
            # The odd one must have an odd prime factor, so its LPF is at least 3.
            # So the minimum possible max is at least 3.
            # If we find 3, we can stop? 
            # Wait, if one is 2-smooth (power of 2) and the other is 3-smooth, max is 3.
            # Is it possible to have max=2? No, because sum is odd.
            # So min possible is 3.
            if best_val == 3:
                break
            if best_val == 2:
                break

    return [best_a, best_b]
