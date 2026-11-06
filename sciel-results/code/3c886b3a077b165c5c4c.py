import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    LIMIT = 1000000
    N = LIMIT + 1  # we need pp up to N (for n+1 when n=LIMIT)
    
    # 1. Compute powerful part for each number using vectorized NumPy approach
    # pp[i] = product of p^e where e>=2 in factorization of i
    # Start with all 1s
    pp = np.ones(N + 1, dtype=np.int64)
    
    # For each prime p, for each power p^k with k>=2, multiply pp[j] by p^k
    # for all j that are divisible by p^k
    # This is equivalent to: for each j, pp[j] = product of p^e where p^e || j and e>=2
    
    # We need to find all primes up to N
    # Use a simple sieve to find primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime, compute powers p^2, p^3, ... up to N
    # For each power p^k (k>=2), multiply pp at indices divisible by p^k by p^k
    # But we need to be careful: if a number is divisible by p^3, it should get p^3, not p^2 * p^3
    # Actually, the powerful part is the product of FULL prime powers p^e where e>=2
    # So for each prime p, we find the maximum e such that p^e divides the number, 
    # and if e>=2, we multiply by p^e
    
    # Better approach: for each prime p, for each multiple m of p^2, p^3, etc.,
    # we need to determine the exact exponent of p in m, and if it's >=2, multiply pp[m] by p^e
    
    # Vectorized approach:
    # For each prime p:
    #   For k = 2, 3, 4, ... while p^k <= N:
    #     For all j divisible by p^k but NOT by p^(k+1):
    #       pp[j] *= p^k
    
    # This is still complex. Let's use a different vectorized approach:
    # For each prime p, compute p^2, p^3, ...
    # For each power p^k, mark all multiples of p^k
    # But we need the exact exponent.
    
    # Alternative: Use the fact that pp[n] = n / squarefree_part(n)
    # where squarefree_part(n) is the product of primes with odd exponent... no that's not right.
    
    # Let's think again. The powerful part is the product of p^e where e>=2.
    # So if n = p1^e1 * p2^e2 * ... * pk^ek, then pp[n] = product of pi^ei for all i where ei>=2.
    
    # Vectorized approach using the sieve method:
    # Initialize pp to 1
    # For each prime p:
    #   Compute p^2, p^3, ... up to N
    #   For each power p^k (k>=2):
    #     For all j that are multiples of p^k but not p^(k+1):
    #       pp[j] *= p^k
    
    # This requires knowing which numbers are divisible by p^k but not p^(k+1).
    
    # Simpler vectorized approach:
    # For each prime p:
    #   Let p2 = p*p
    #   For all j divisible by p2:
    #     Determine the exact exponent of p in j
    #     If exponent >= 2, multiply pp[j] by p^exponent
    
    # Even simpler: use the SPF approach but vectorized
    # Actually, let's just use the pure Python SPF approach since it's fast enough for N=1M
    
    # Pure Python SPF sieve
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    if time.time() - start_time > DEADLINE - 30:
        # Not enough time
        pass
    
    # Compute powerful part using SPF
    pp_list = [1] * (N + 1)
    
    for i in range(2, N + 1):
        x = i
        res = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                res *= (p ** e)
        pp_list[i] = res
        
    if time.time() - start_time > DEADLINE - 10:
        # Time check
        pass
    
    # 3. Find n that maximizes pp[n] * pp[n+1]
    pp_arr = np.array(pp_list[10000:N+1], dtype=np.int64)
    
    if len(pp_arr) < 2:
        return 10000
    
    products = pp_arr[:-1] * pp_arr[1:]
    
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    return int(best_n)
