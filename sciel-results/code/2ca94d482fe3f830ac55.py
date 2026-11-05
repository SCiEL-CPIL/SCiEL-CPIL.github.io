import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0

    def get_primes_up_to(limit):
        if limit < 2:
            return []
        is_prime = [True] * (limit + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, val in enumerate(is_prime) if val]

    def generate_smooth_numbers(primes, limit):
        if not primes:
            return {1}
        
        smooth = {1}
        for p in primes:
            new_smooth = set()
            current_smooth = list(smooth)
            for s in current_smooth:
                val = s
                while val * p <= limit:
                    val *= p
                    new_smooth.add(val)
            smooth.update(new_smooth)
        return smooth

    def exists_pair(K):
        primes = get_primes_up_to(K)
        if not primes:
            if N == 2:
                return True, 1, 1
            return False, 0, 0
            
        smooth_set = generate_smooth_numbers(primes, N)
        for a in smooth_set:
            b = N - a
            if b in smooth_set:
                return True, a, b
        return False, 0, 0

    # Linear search for smallest K
    # We know K=11 works. We check 1, 2, 3, 5, 7, 11...
    # Primes: 2, 3, 5, 7, 11, 13, ...
    # We can just iterate K from 1 to N, but skip non-prime K?
    # No, K is the bound on prime factors.
    # If K is not prime, the set of primes <= K is the same as for the largest prime < K.
    # So we only need to check K being a prime (or 1).
    
    candidates = [1] + get_primes_up_to(N)
    
    for K in candidates:
        if time.time() > deadline:
            break
        exists, a, b = exists_pair(K)
        if exists:
            return [a, b]
            
    # Fallback
    return [1, N-1]
