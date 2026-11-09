import math

def construct():
    # Sieve of Eratosthenes for primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    best_k = 1
    best_s = 1
    target = 200
    
    # Systematic scan from high end where collisions are rare
    # Check consecutive blocks of 200 numbers starting from 10^12 - 200
    # Going downward, stop when we find k=200
    
    # Check up to 50 blocks of 200 numbers each
    for block_idx in range(50):
        s = 10**12 - (block_idx + 1) * 200
        if s < 1:
            s = 1
        
        seen = set()
        k = 0
        for i in range(200):
            n = s + i
            if n > 10**12:
                break
            val = phi(n)
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= target:
                break
    
    return {'start': best_s, 'length': best_k}
