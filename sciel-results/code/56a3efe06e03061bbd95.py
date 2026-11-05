"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    # Generate primes up to 10^6 using sieve
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute smallest prime factor for numbers up to 10^6
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def phi_small(n):
        """Compute phi(n) for n <= 10^6 using SPF table."""
        if n <= 1:
            return n
        result = n
        y = n
        while y > 1:
            p = spf[y]
            result -= result // p
            while y % p == 0:
                y //= p
        return result
    
    def phi_large(n):
        """Compute phi(n) for n up to 10^12 by trial division with primes."""
        if n <= limit:
            return phi_small(n)
        result = n
        y = n
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
    
    # Strategy: Search in windows around multiples of primorials
    # Primorial: product of first few primes
    # This gives numbers with many small prime factors, making phi values
    # more likely to be distinct
    
    # Generate primorial-based starting points
    primorial = 1
    for p in primes[:10]:  # first 10 primes
        primorial *= p
    
    best_k = 1
    best_s = 1
    
    # First, do a quick search in small range to establish baseline
    # Then search in structured regions
    
    # Phase 1: Quick scan of small numbers (fast, establishes baseline)
    window_size = 200
    # Compute phi for 1..5000
    phi_cache = [0] * 5001
    for i in range(1, 5001):
        phi_cache[i] = phi_small(i)
    
    for s in range(1, 4801):
        seen = set()
        k = 0
        while s + k <= 5000 and phi_cache[s + k] not in seen:
            seen.add(phi_cache[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return {'start': best_s, 'length': best_k}
    
    # Phase 2: Search in structured regions near multiples of primorial
    # Try windows centered at multiples of primorial
    step = primorial // 10  # Use a fraction of primorial for step size
    
    # Also try random large numbers
    random.seed(42)
    
    # Try several starting points
    candidates = []
    # Multiples of primorial
    for mult in range(1, 20):
        s = mult * primorial
        if s <= 10**12 - window_size:
            candidates.append(s)
    
    # Random large numbers
    for _ in range(100):
        s = random.randint(1, 10**12 - window_size)
        candidates.append(s)
    
    # Sort candidates and try them
    candidates = list(set(candidates))
    random.shuffle(candidates)
    
    for s in candidates:
        if time.time() > deadline - 5:
            break
        
        # Compute phi for window [s, s+199]
        phi_vals = []
        valid = True
        for i in range(window_size):
            n = s + i
            if n > 10**12:
                valid = False
                break
            phi_vals.append(phi_large(n))
        
        if not valid:
            continue
        
        # Find longest prefix with distinct phi values
        seen = set()
        k = 0
        for v in phi_vals:
            if v in seen:
                break
            seen.add(v)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
