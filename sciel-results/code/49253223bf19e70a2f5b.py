"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    TARGET = 123457

    # Precompute largest prime factor for all n up to TARGET
    # lpf[n] will store the largest prime factor of n
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1  # P(1) = 1

    # Sieve of Eratosthenes variant to find largest prime factor
    # For each prime p, update lpf for all multiples of p
    # We iterate p from 2 to TARGET
    # Optimization: Only need to go up to sqrt(TARGET) for trial division if we were factoring individually,
    # but here we are doing a sieve.
    # Standard approach:
    # Initialize lpf array to 0.
    # For i from 2 to TARGET:
    #   if lpf[i] == 0: # i is prime
    #      for j from i to TARGET step i:
    #          lpf[j] = i  # i is a prime factor of j. Since we iterate i in increasing order,
    #                       # the last prime factor assigned will be the largest?
    #                       # No, if we do lpf[j] = i, and i increases, then yes, the largest prime factor will overwrite smaller ones.
    #                       # Example: j=12.
    #                       # i=2: lpf[12]=2
    #                       # i=3: lpf[12]=3
    #                       # i=5: no
    #                       # i=7: no
    #                       # Final lpf[12]=3. Correct.
    
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i

    best_val = float('inf')
    best_pair = (1, TARGET - 1)

    # Iterate through all possible a
    # We can stop early if we find val=2 (only possible if both are powers of 2, sum=123457 odd, impossible)
    # Min possible val is likely small.
    for a in range(1, TARGET):
        if time.time() > deadline - 1.0:
            break
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        val = pa if pa > pb else pb
        if val < best_val:
            best_val = val
            best_pair = (a, b)
            # Early exit if we hit a theoretical lower bound?
            # P(n) >= 2 for n>1. 
            # If best_val == 2, we can't do better unless one is 1.
            # If a=1, b=123456. P(1)=1, P(123456)=? 
            # 123456 = 2^4 * 3 * 11 * 29. P=29. Max(1,29)=29.
            # If we find val=2, it means both are powers of 2. Sum of two powers of 2 is odd? No, 2^a+2^b is even if a,b>=1.
            # 123457 is odd. So one must be odd. Odd numbers > 1 have P >= 3.
            # So min val is at least 3?
            # If one is 1, P(1)=1. Other is 123456. P(123456)=29. Max=29.
            # If we find val=3, it's very good.
            if best_val <= 3:
                # Can we get 2? No, as argued.
                # Can we get 3? Both must have factors <= 3.
                # a = 2^x 3^y, b = 2^u 3^v.
                # Sum is odd, so one is odd (power of 3), one is even.
                # If we find 3, it's likely optimal or near-optimal.
                # We can continue to be sure, but 3 is very low.
                pass

    a, b = best_pair
    if time.time() > deadline:
        return [1, TARGET - 1]
    return [a, b]
# EVOLVE-BLOCK-END
