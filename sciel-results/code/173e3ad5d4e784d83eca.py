"""Rewritten program using segmented sieve for SPF over [p^2+1, p^2+200]."""
# EVOLVE-BLOCK-START
import math

def construct():
    # Find prime p <= 1e9 and k (1..200) maximizing k such that
    # every integer in [p^2+1, p^2+k] has all prime factors <= p.
    
    def sieve_primes(limit):
        """Return list of primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute primes up to 500_000
    max_sieve = 500_000
    primes = sieve_primes(max_sieve)
    
    best_k = 0
    best_p = 0
    
    # For each prime p, check the longest run of p-smooth numbers
    for p in primes:
        p2 = p * p
        
        k = 0
        for offset in range(1, 201):
            n = p2 + offset
            temp = n
            smooth = True
            # Divide by all primes <= p
            for q in primes:
                if q > p:
                    break
                if q * q > temp:
                    break
                while temp % q == 0:
                    temp //= q
            if temp > 1 and temp > p:
                smooth = False
                break
            if not smooth:
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break
    
    if best_k >= 1:
        return {'p': best_p, 'k': best_k}
    
    return {'p': 3, 'k': 1}

# EVOLVE-BLOCK-END
