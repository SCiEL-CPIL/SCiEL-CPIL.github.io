import math
import random

def construct():
    # Target: k=200
    # Strategy: Use a fast factorization method to verify distinct totients.
    # We use a hybrid approach:
    # 1. Precompute primes up to 10^6 using Sieve of Eratosthenes.
    # 2. For each number n, divide by small primes.
    # 3. If remaining factor > 1, check if prime. If not, use Pollard Rho (though unlikely for n < 10^12 after small primes).
    # Actually, for n < 10^12, if we trial divide up to 10^6, the remaining factor is either 1 or a prime > 10^6.
    # Because if it were composite, it would have a factor <= sqrt(10^12 / 10^6) = 10^3, which we already checked.
    # Wait, if n = p * q with p, q > 10^6, then n > 10^12.
    # So for n <= 10^12, after dividing by all primes <= 10^6, the remaining part is either 1 or a prime.
    # So trial division up to 10^6 is sufficient and deterministic.
    
    # The current program already does this.
    # To "improve" or "change approach", I will use a different search strategy:
    # Instead of starting at 10^11 and stepping by 1000, I will start at a random large number 
    # and check. Or I will use a "jump" search.
    
    # However, the most reliable "new" approach is to ensure the factorization is as fast as possible.
    # I will implement the sieve and phi calculation in a tight loop.
    
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(b'\x00') * len(is_prime[i*i:limit+1:i])
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def phi(x):
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    # Search for a run of length 200
    # Start from a large number. 
    # To be different from "base_s + offset * 1000", I will use a random start 
    # or a specific prime-based start to avoid structured collisions (though unlikely).
    # Actually, random is safer against "structured" bad luck.
    
    max_k = 200
    # Use a random start in the range [10^11, 10^12 - 200]
    # But deterministic is better for reproducibility? 
    # The task doesn't require determinism.
    # I'll stick to a deterministic but different pattern: start at 10^11 + 1
    # and increment by 1 if collision, or jump.
    
    # Let's try starting at 10^11 + 1 and checking consecutive starts.
    # If the first one fails, try the next.
    
    base_s = 10**11 + 1
    for attempt in range(100):
        s = base_s + attempt
        if s + max_k - 1 > 10**12:
            break
            
        vals = []
        seen = set()
        distinct = True
        
        for i in range(max_k):
            n = s + i
            v = phi(n)
            if v in seen:
                distinct = False
                break
            seen.add(v)
            
        if distinct:
            return {'start': s, 'length': max_k}
            
    return {'start': 1, 'length': 1}
