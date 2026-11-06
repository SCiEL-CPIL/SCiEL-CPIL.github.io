import math
import time

def construct():
    deadline = time.time() + 140  # 140s safety margin
    
    def sieve(n):
        """Generate primes up to n using sieve of Eratosthenes."""
        if n < 2:
            return []
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    # Generate primes up to a reasonable bound
    # We'll try primes up to 10^6 first, then extend if time allows
    max_p = 10**6
    primes = sieve(max_p)
    
    best_k = 0
    best_p = 2
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # Check how many consecutive values p²+1, p²+2, ... are p-smooth
        k = 0
        for j in range(1, 201):
            n = p * p + j
            # Check if n is p-smooth: all prime factors ≤ p
            temp = n
            # Trial division by primes ≤ p
            smooth = True
            for q in primes:
                if q > p:
                    break
                if q * q > temp:
                    break
                while temp % q == 0:
                    temp //= q
            if temp > 1 and temp > p:
                smooth = False
            if not smooth:
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we found k=200, we can stop
            if best_k == 200:
                break
    
    return {'p': best_p, 'k': best_k}
