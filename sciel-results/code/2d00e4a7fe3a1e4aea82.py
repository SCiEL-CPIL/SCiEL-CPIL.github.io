"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def _sieve_primes(n):
    """Return list of primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _powerful_part(n, primes):
    """Compute the powerful part of n: product of p^e where e >= 2."""
    prod = 1
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                prod *= p ** e
    if n > 1:
        # n is prime with exponent 1, doesn't contribute
        pass
    return prod

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    LO = 10000
    HI = 1000000
    
    # Get primes up to sqrt(1000000) ~ 1000, plus some larger ones for powerful parts
    primes = _sieve_primes(2000)
    
    # Strategy: Use CRT to find n such that:
    # n ≡ 0 (mod p^e) for some prime p with e >= 2
    # n+1 ≡ 0 (mod q^f) for some prime q with f >= 2
    # where p ≠ q
    
    # First, do a fast sieve-based scan to find good candidates
    # Then use CRT to construct better candidates
    
    # Fast scan using multiplicative sieve
    N = HI + 1
    pp = [1] * (N + 1)
    
    # For each prime p, for each power p^e with e >= 2, 
    # multiply pp[m] by p^e for all multiples m of p^e
    for p in primes:
        if p * p > N:
            break
        power = p * p
        while power <= N:
            for m in range(power, N + 1, power):
                pp[m] *= power
            power *= p
    
    # Find best from scan
    best_n = LO
    best_val = pp[LO] * pp[LO + 1]
    for n in range(LO, HI + 1):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
        if time.time() > deadline:
            break
    
    # Now try CRT-based construction for potentially better results
    # We want n ≡ 0 (mod a) and n+1 ≡ 0 (mod b) where a = p^e, b = q^f
    # This means n ≡ 0 (mod a) and n ≡ -1 (mod b)
    # By CRT, solution exists iff gcd(a, b) = 1, i.e., p ≠ q
    
    # Try combinations of prime powers
    prime_powers = []
    for p in primes:
        if p > 100:  # limit to reasonable primes
            break
        power = p * p
        while power <= HI:
            prime_powers.append((p, power))
            power *= p
    
    # Sort by potential contribution (larger prime powers first)
    prime_powers.sort(key=lambda x: -x[1])
    
    # Try top combinations
    tried = set()
    improved = False
    for (p1, a), (p2, b) in combinations(prime_powers[:50], 2):
        if p1 == p2:
            continue
        if time.time() > deadline:
            break
        
        # Solve n ≡ 0 (mod a), n ≡ -1 (mod b)
        # n = k*a, and k*a ≡ -1 (mod b)
        # k ≡ -a^{-1} (mod b)
        
        # Find modular inverse of a mod b
        try:
            # Extended Euclidean algorithm
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x, y = egcd(b, a % b)
                return (g, y, x - (a // b) * y)
            
            g, x, y = egcd(a, b)
            if g != 1:
                continue  # not coprime
            
            # k ≡ -x (mod b)
            k0 = (-x) % b
            # n = k*a = (k0 + t*b)*a for t = 0, 1, 2, ...
            # Find smallest n in [LO, HI]
            
            lcm = a * b  # since gcd(a,b)=1
            
            # n = k0*a + t*lcm
            # Find t such that LO <= n <= HI
            base = k0 * a
            if base < LO:
                t = (LO - base + lcm - 1) // lcm
            else:
                t = 0
            
            n = base + t * lcm
            if n > HI:
                continue
            
            # Check this candidate and nearby ones
            for delta in range(-10, 11):
                cand = n + delta
                if LO <= cand <= HI:
                    v = pp[cand] * pp[cand + 1]
                    if v > best_val:
                        best_val = v
                        best_n = cand
                        improved = True
        except Exception:
            continue
    
    return best_n
# EVOLVE-BLOCK-END
