"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import heapq

def construct():
    """Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120."""
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # We want to find the minimum n such that phi(n) % 120 == 0
    # phi(n) = n * prod_{p|n} (1 - 1/p)
    # 120 = 2^3 * 3 * 5
    # 
    # Strategy: iterate n from 1 upward, compute phi(n), check if divisible by 120
    # For efficiency, we can use a sieve-like approach to compute phi for all n up to some limit
    # But since we want the MINIMUM n, we can just iterate and compute phi for each n
    # 
    # Better: use a precomputed sieve for phi up to a reasonable limit
    # The answer is likely small, so let's try a direct approach first
    
    # Let's think about what n works:
    # phi(n) must be divisible by 120 = 8 * 3 * 5
    # 
    # For prime p, phi(p) = p-1
    # For p^k, phi(p^k) = p^k - p^(k-1) = p^(k-1)*(p-1)
    # 
    # Let's try small values:
    # n=1: phi=1, 1%120 != 0
    # n=2: phi=1, no
    # n=3: phi=2, no
    # ...
    # 
    # Let's use a more systematic approach. We'll compute phi(n) for increasing n
    # using trial division for each n, which is O(sqrt(n)) per n
    # 
    # Actually, let's just iterate and compute phi efficiently
    
    def totient(n):
        """Compute Euler's totient function phi(n)"""
        if n <= 1:
            return n
        result = n
        x = n
        p = 2
        while p * p <= x:
            if x % p == 0:
                while x % p == 0:
                    x //= p
                result -= result // p
            p += 1
        if x > 1:
            result -= result // x
        return result
    
    # Iterate n from 1 upward
    for n in range(1, 1000001):
        if time.time() > deadline:
            # Return the best we can - but we haven't found one yet
            # This shouldn't happen for small n
            break
        phi_n = totient(n)
        if phi_n % 120 == 0:
            return n
    
    # Fallback: return a known valid answer
    # phi(120) = 120 * (1-1/2) * (1-1/3) * (1-1/5) = 120 * 1/2 * 2/3 * 4/5 = 32
    # That's not divisible by 120.
    # phi(180) = 180 * (1-1/2) * (1-1/3) * (1-1/5) = 180 * 1/2 * 2/3 * 4/5 = 48
    # Not divisible by 120 either.
    # phi(240) = 240 * 1/2 * 2/3 * 4/5 = 64
    # phi(360) = 360 * 1/2 * 2/3 * 4/5 = 96
    # phi(600) = 600 * 1/2 * 2/3 * 4/5 = 160
    # phi(1200) = 1200 * 1/2 * 2/3 * 4/5 = 320
    # 
    # Let's think differently. We need phi(n) divisible by 120.
    # phi(n) is even for n > 2.
    # 
    # Let's try n = 2^a * 3^b * 5^c * ... 
    # phi(2^a) = 2^(a-1) for a >= 1
    # phi(3^b) = 2 * 3^(b-1)
    # phi(5^c) = 4 * 5^(c-1)
    # 
    # For n = 2^a * 3^b * 5^c:
    # phi(n) = 2^(a-1) * 2 * 3^(b-1) * 4 * 5^(c-1) = 2^(a+1) * 3^(b-1) * 5^(c-1)
    # 
    # We need 2^3 * 3 * 5 | phi(n)
    # So: 2^(a+1) must have at least 2^3, so a+1 >= 3, i.e., a >= 2
    # 3^(b-1) must have at least 3^1, so b-1 >= 1, i.e., b >= 2
    # 5^(c-1) must have at least 5^1, so c-1 >= 1, i.e., c >= 2
    # 
    # So n = 2^a * 3^b * 5^c with a>=2, b>=2, c>=2
    # Minimum such n = 2^2 * 3^2 * 5^2 = 4 * 9 * 25 = 900
    # phi(900) = 900 * (1-1/2) * (1-1/3) * (1-1/5) = 900 * 1/2 * 2/3 * 4/5 = 240
    # 240 % 120 == 0! So n=900 works.
    # 
    # But is there a smaller n? What if n has other prime factors?
    # If n includes a prime p > 5, then phi(n) includes (p-1) as a factor (times other terms)
    # 
    # Let's check if there's a smaller n. We need to be more careful.
    # The above analysis assumes n only has prime factors 2, 3, 5.
    # But n could have other primes.
    # 
    # For example, if n = 2^a * p where p is prime > 5:
    # phi(n) = 2^(a-1) * (p-1)
    # We need 120 | 2^(a-1) * (p-1)
    # 120 = 8 * 3 * 5
    # So we need 8 | 2^(a-1) * (p-1), 3 | 2^(a-1) * (p-1), 5 | 2^(a-1) * (p-1)
    # Since gcd(2,3)=1 and gcd(2,5)=1, we need:
    # 3 | (p-1) and 5 | (p-1) if a-1 < 3, or more generally 3*5/gcd(3*5, 2^(a-1)) | (p-1)
    # Since 2^(a-1) is a power of 2, gcd(15, 2^(a-1)) = 1, so we need 15 | (p-1)
    # And 8/gcd(8, 2^(a-1)) | (p-1) ... wait, this is getting complicated.
    # 
    # Let's just trust the loop. n=900 should be found, and maybe something smaller.
    # Let's return 900 as a fallback if the loop doesn't find anything smaller.
    
    return 900
# EVOLVE-BLOCK-END
