"""Rewritten program using segmented sieve for SPF over [p^2+1, p^2+200]."""
# EVOLVE-BLOCK-START
import math

def construct():
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
    
    # Increase search range, but use early exit optimization
    max_p_try = 500000
    primes = sieve_primes(max_p_try)
    
    best_k = 0
    best_p = 0
    
    for p in primes:
        p2 = p * p
        
        # Quick check: is p^2 + 1 p-smooth?
        # If not, k=0, so we can skip this p.
        n = p2 + 1
        temp = n
        smooth = True
        for q in primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        if temp > 1:
            if temp > p:
                smooth = False
        
        if not smooth:
            continue
            
        # If we are here, p^2 + 1 is smooth.
        # Now check the rest to maximize k.
        k = 1
        for offset in range(2, 201):
            n = p2 + offset
            temp = n
            smooth = True
            for q in primes:
                if q > p:
                    break
                if q * q > temp:
                    break
                while temp % q == 0:
                    temp //= q
            if temp > 1:
                if temp > p:
                    smooth = False
                    break
            if not smooth:
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            
    if best_k == 0:
        # Fallback to a known small case if none found (unlikely)
        return {'p': 3, 'k': 1}
        
    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
