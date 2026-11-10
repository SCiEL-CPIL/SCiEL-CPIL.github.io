import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 1000000
    max_val = N + 1
    
    # SPF sieve
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful parts
    pp = [1] * (max_val + 1)
    
    for i in range(2, max_val + 1):
        if spf[i] == i: # i is prime
            # This is not the right way. We need to compute P(i) for all i.
            pass
            
    # Better: iterate through numbers and factorize using SPF
    # But doing this for each i is O(N log N).
    # We can compute P(i) recursively or iteratively.
    
    # Let's use the sieve method from before, it was correct.
    # I will just copy the previous logic but ensure no bugs.
    
    pp = [1] * (max_val + 1)
    limit = int(math.isqrt(max_val))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    for p in primes:
        p2 = p * p
        if p2 > max_val:
            continue
        
        max_k = max_val // p2
        
        for k in range(1, max_k + 1):
            m = k * p2
            temp = k
            extra_exp = 0
            while temp % p == 0:
                temp //= p
                extra_exp += 1
            
            total_exp = 2 + extra_exp
            pp[m] *= p ** total_exp
        
        if time.time() > deadline:
            break
    
    best_n = 10000
    max_prod = 0
    
    pp_local = pp
    for n in range(10000, N + 1):
        prod = pp_local[n] * pp_local[n + 1]
        if prod > max_prod:
            max_prod = prod
            best_n = n
        
        if (n - 10000) % 10000 == 0 and time.time() > deadline:
            break
    
    return best_n
