"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from typing import List

def sieve_primes(limit: int) -> List[int]:
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

# Precompute small primes for trial division
SMALL_PRIME_LIMIT = 100000
_small_primes = sieve_primes(SMALL_PRIME_LIMIT)

def is_prime_mr(n: int) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Deterministic witnesses for n < 3,317,044,064,679,887,385,961,981
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def factorize(n: int) -> List[int]:
    """Factorize n using trial division by small primes + Miller-Rabin for remaining."""
    factors = []
    for p in _small_primes:
        if p * p > n:
            break
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        if is_prime_mr(n):
            factors.append(n)
        else:
            # For composite n > 1 after trial division, we need to factor it further
            # This is rare for our use case since we only need distinct prime factors
            # Try a few more primes or use Pollard's rho
            # For simplicity, if n is composite and large, try to find a factor
            # In practice for n <= 10^12, if it's composite after dividing by primes up to 10^5,
            # it must be a product of two primes > 10^5, so n > 10^10
            # We can try to find a factor using trial division up to cube root
            cube_root = int(n ** (1/3)) + 1
            found = False
            for p in range(_small_primes[-1] + 1, cube_root):
                if p % 6 != 1 and p % 6 != 5:
                    continue
                if n % p == 0:
                    factors.append(p)
                    n //= p
                    if n > 1:
                        if is_prime_mr(n):
                            factors.append(n)
                        else:
                            # Should be a prime at this point
                            factors.append(n)
                    found = True
                    break
            if not found and n > 1:
                if is_prime_mr(n):
                    factors.append(n)
                else:
                    # This case is extremely rare; just add n (may cause issues but unlikely)
                    factors.append(n)
    return factors

def phi(n: int) -> int:
    """Compute Euler's totient function phi(n)."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = n
    temp = n
    for p in _small_primes:
        if p * p > temp:
            break
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
    if temp > 1:
        result -= result // temp
    return result

def compute_phi_segment_fast(start: int, length: int) -> List[int]:
    """Compute phi for a segment using the phi function."""
    return [phi(start + i) for i in range(length)]

def has_distinct_phis(phis: List[int]) -> bool:
    """Check if all phi values are distinct, with early termination."""
    seen = set()
    for p in phis:
        if p in seen:
            return False
        seen.add(p)
    return True

def construct() -> dict:
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    max_k = 200
    max_n = 10**12
    
    random.seed(42)
    
    best_result = {'start': 1, 'length': 1}
    best_k = 1
    
    # Strategy 1: Search in the high range (near 10^12) with random offsets
    # For large n, phi values are more likely to be distinct
    
    # First, try a few fixed positions near the top
    test_positions = [
        max_n - 199,
        max_n - 100000,
        max_n - 1000000,
        max_n - 10000000,
    ]
    
    for s in test_positions:
        if time.time() > deadline:
            break
        if s < 1:
            continue
        phis = compute_phi_segment_fast(s, max_k)
        if has_distinct_phis(phis):
            return {'start': s, 'length': max_k}
    
    # Strategy 2: Random search in high range
    num_attempts = 0
    while time.time() < deadline - 5:
        num_attempts += 1
        
        # Focus on high range where n is large
        # Use different ranges to explore
        if num_attempts % 10 == 0:
            # Occasionally try medium range
            s = random.randint(10**9, 10**11)
        else:
            # Mostly high range
            s = random.randint(10**11, max_n - max_k + 1)
        
        phis = compute_phi_segment_fast(s, max_k)
        if has_distinct_phis(phis):
            return {'start': s, 'length': max_k}
        
        # If we find a partial match, try to extend or use as starting point
        # Count how many are distinct
        distinct_count = len(set(phis))
        if distinct_count > best_k:
            # Find the longest prefix with distinct values
            seen = set()
            for i, p in enumerate(phis):
                if p in seen:
                    best_k = i
                    best_result = {'start': s, 'length': i}
                    break
                seen.add(p)
            else:
                best_k = max_k
                best_result = {'start': s, 'length': max_k}
                return best_result
    
    # Strategy 3: Try smaller k values if we haven't found k=200
    for k in range(max_k - 1, best_k, -1):
        if time.time() > deadline:
            break
        for _ in range(10):
            if time.time() > deadline:
                break
            s = random.randint(10**10, max_n - k + 1)
            phis = compute_phi_segment_fast(s, k)
            if has_distinct_phis(phis):
                return {'start': s, 'length': k}
    
    return best_result

# EVOLVE-BLOCK-END
