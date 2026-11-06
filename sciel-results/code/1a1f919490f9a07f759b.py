import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85
    
    N = 1000000
    pp = [1] * (N + 2)
    
    # Generate primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    # For each prime p, for each multiple m of p, compute v_p(m) and if >=2, multiply pp[m] by p^e
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # For multiples of p^2, p^3, etc.
        # Better: iterate multiples of p, track power
        # Even better: for each power pe = p^k (k>=2), for multiples of pe not divisible by p^(k+1)
        pe = p2
        while pe <= N:
            pe_next = pe * p
            # Multiples of pe that are NOT multiples of pe_next
            for m in range(pe, N+1, pe):
                if m % pe_next != 0:
                    pp[m] *= pe
            pe = pe_next
            if time.time() - start_time > DEADLINE - 10:
                break
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Find best consecutive pair
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    for n in range(10000, N):
        val = pp[n] * pp[n+1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
