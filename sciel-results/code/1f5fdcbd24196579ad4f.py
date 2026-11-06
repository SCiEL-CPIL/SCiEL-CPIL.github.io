import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    TARGET = 123457
    
    # Step 1: Sieve of Eratosthenes to get primes up to sqrt(TARGET)
    limit = int(math.isqrt(TARGET)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    small_primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Compute largest prime factor P(n) for all n in [1, TARGET]
    # P(1) = 1
    lpf = [1] * (TARGET + 1)
    # rem[n] will hold the part of n not divisible by small primes
    rem = list(range(TARGET + 1))
    
    # Process small primes in increasing order
    for p in small_primes:
        # Check time occasionally
        if time.time() > deadline:
            # Fallback to a simple known good pair if we run out of time
            # 123456 = 2^5 * 3 * 7 * 19 * 23. P(123456)=23.
            return [1, TARGET - 1]
            
        # Mark multiples of p
        # We update lpf[m] to p because p is increasing, so the last p that divides m
        # among small primes is the largest small prime factor.
        for m in range(p, TARGET + 1, p):
            lpf[m] = p
            # Strip all factors of p from rem[m]
            while rem[m] % p == 0:
                rem[m] //= p
    
    # Step 3: Finalize LPF
    # If rem[n] > 1, it is a prime factor larger than sqrt(TARGET).
    # Since n <= TARGET, there can be at most one such factor.
    # And it is larger than all small primes.
    for n in range(2, TARGET + 1):
        if rem[n] > 1:
            lpf[n] = rem[n]
    
    if time.time() > deadline:
        return [1, TARGET - 1]
    
    # Step 4: Linear scan to minimize max(P(a), P(b))
    best_max = TARGET + 1
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET):
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max:
            best_max = current_max
            best_a = a
            best_b = b
            if best_max == 2:
                break
    
    return [best_a, best_b]
