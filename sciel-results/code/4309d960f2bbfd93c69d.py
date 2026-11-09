import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # need n+1 up to 1000001
    
    # powerful_part[i] = product of p^e for all prime powers p^e (e>=2) dividing i
    powerful_part = [1] * (MAX_VAL + 1)
    
    # Sieve: for each prime p, for each power p^e with e>=2,
    # multiply p^e into powerful_part[m] for all m divisible by p^e
    # But we need the FULL p^e that divides m, not just p^2.
    # Actually, if p^e divides m, then the full power of p in m is p^f where f>=e.
    # The "full prime power" means: if p^f || m (f is the exact exponent), then
    # if f>=2, we include p^f in the powerful part.
    
    # Better approach: for each number, factorize and compute powerful part.
    # But that's O(N * sqrt(N)) which is too slow.
    
    # Use SPF (smallest prime factor) sieve, then for each number,
    # compute its factorization and the powerful part.
    
    # Build SPF table
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each number, compute powerful part using SPF
    # powerful_part[n] = product of p^e where e>=2 in factorization of n
    pow_part = [1] * (MAX_VAL + 1)
    
    for i in range(2, MAX_VAL + 1):
        if time.time() > deadline - 5.0:
            break
        n = i
        pp = 1
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        pow_part[i] = pp
    
    # Now find n in [10000, 1000000] that maximizes pow_part[n] * pow_part[n+1]
    best_n = 10000
    best_val = pow_part[10000] * pow_part[10001]
    
    for n in range(10000, N + 1):
        if time.time() > deadline - 2.0:
            break
        val = pow_part[n] * pow_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
