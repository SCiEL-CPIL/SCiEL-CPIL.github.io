import math

def construct():
        # 1. Sieve
        limit = 10**6
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        primes = [i for i, v in enumerate(is_prime) if v]

        # 2. Phi
        def phi(x):
            if x == 1: return 1
            result = x
            temp = x
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    result -= result // p
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                result -= result // temp
            return result

        # 3. Search
        # We want max k. Max allowed is 200.
        # Try k from 200 down to 1.
        for k in range(200, 0, -1):
            # Try a few candidate starts
            # Max start is 10^12 - k + 1
            candidates = [
                10**12 - k + 1,
                10**11,
                10**9,
                10**6,
                10**4
            ]
            # Filter valid starts
            candidates = [s for s in candidates if s >= 1 and s + k - 1 <= 10**12]
            
            for s in candidates:
                vals = [phi(s + i) for i in range(k)]
                if len(set(vals)) == k:
                    return {'start': s, 'length': k}
        
        # Fallback (should not happen)
        return {'start': 1, 'length': 1}
