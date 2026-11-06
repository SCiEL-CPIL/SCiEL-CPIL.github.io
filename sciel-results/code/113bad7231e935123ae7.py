import math
import bisect

def construct():
    N = 123457
    
    # Sieve of Eratosthenes to find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def get_smooth_numbers(K):
        idx = bisect.bisect_right(primes, K)
        primes_K = primes[:idx]
        
        current = [1]
        for p in primes_K:
            new_nums = []
            for c in current:
                val = c
                while val * p <= N:
                    val *= p
                    new_nums.append(val)
            current.extend(new_nums)
        return set(current)
    
    def check(K):
        S = get_smooth_numbers(K)
        for a in S:
            if N - a in S:
                return True, a
        return False, None

    lo, hi = 2, N
    best_a = 1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, a_val = check(mid)
        if ok:
            best_a = a_val
            hi = mid - 1
        else:
            lo = mid + 1
            
    return [best_a, N - best_a]
