import time
from math import isqrt

# Global deadline
_DEADLINE = 105.0
_start = time.perf_counter()

def _time_left():
    return _DEADLINE - (time.perf_counter() - _start)

def _sieve(n):
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            is_prime[i*i:n+1:i] = bytearray(len(is_prime[i*i:n+1:i])) * 0
    return [i for i, val in enumerate(is_prime) if val]

def construct():
    global _start
    _start = time.perf_counter()
    
    # Sieve primes up to a limit. 
    # 2,000,000 primes takes < 0.1s to sieve.
    # There are ~150k primes.
    SIEVE_LIMIT = 2000000
    primes = _sieve(SIEVE_LIMIT)
    
    best_p = 3
    best_k = 1
    
    # Iterate through candidate primes
    # We can optimize by skipping very small p if we know they don't beat current best, 
    # but k=1 is the baseline. We want to find k > 1.
    
    for p in primes:
        if _time_left() < 2.0:
            break
            
        L = p * p + 1
        R = L + 200  # We check p^2+1 to p^2+200
        
        # rem[i] corresponds to L + i
        rem = list(range(L, R + 1))
        
        # Divide out all prime factors <= p
        for d in primes:
            if d > p:
                break
            
            # Find the first multiple of d in [L, R]
            start = ((L + d - 1) // d) * d
            if start > R:
                continue
                
            for j in range(start, R + 1, d):
                idx = j - L
                # Divide out all factors of d
                while rem[idx] % d == 0:
                    rem[idx] //= d
                    
        # Count how many are 1 (i.e., p-smooth)
        k = 0
        for val in rem:
            if val == 1:
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
    # If no improvement found, return a valid small case
    # p=3, 3^2+1=10=2*5 (5>3, so not 3-smooth). 
    # Wait, 10 has factor 5 which is > 3. So k=0 for p=3?
    # Let's check p=2. 2^2+1=5 (5>2). k=0.
    # p=5. 5^2+1=26=2*13 (13>5). k=0.
    # p=7. 7^2+1=50=2*5^2. 5<7, 2<7. Smooth.
    # 7^2+2=51=3*17 (17>7). Not smooth.
    # So for p=7, k=1.
    
    if best_k == 0:
        # Find a valid one
        # p=7 gives k=1
        return {'p': 7, 'k': 1}
        
    return {'p': best_p, 'k': best_k}
