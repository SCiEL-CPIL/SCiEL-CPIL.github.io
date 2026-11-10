import time
import math
from itertools import combinations, product

def _tau_perp(n):
    """Compute the number of adjacent coprime pairs among divisors of n."""
    if n < 2:
        return 0
    # Factorize n
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                exp += 1
                temp //= d
            factors.append((d, exp))
        d += 1
    if temp > 1:
        factors.append((temp, 1))
    
    # Generate all divisors
    divisors = [1]
    for p, e in factors:
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start = time.time()
    deadline = start + 140.0  # 140s deadline, 10s safety margin
    
    # Primes up to ~100
    primes = _sieve(200)
    
    # The product of first 15 primes is about 6.1e17
    # First 16 primes product is about 3.2e19 (too big)
    # So max 15 distinct primes with exponent 1
    
    # Strategy: try different combinations of primes and exponents
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # seed
    best_score = _tau_perp(best_n)
    
    # Try squarefree numbers with various prime sets
    # For k primes, product must be <= 10^18
    max_primes = 15
    
    # Try all subsets of size k from first ~20 primes
    # But that's C(20,15) = 15504, manageable
    # Actually, we want to maximize the number of coprime adjacent pairs
    # Let's try various sizes
    
    # First, let's understand: for a squarefree n with k primes,
    # how many coprime adjacent pairs do we get?
    # Let's test a few cases
    
    # Try k=8 (seed)
    seed_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    seed_n = 1
    for p in seed_primes:
        seed_n *= p
    best_score = _tau_perp(seed_n)
    best_n = seed_n
    
    # Try k=14, 15 with smallest primes
    for k in range(8, 16):
        if k > len(primes):
            break
        # Use the k smallest primes
        candidate = 1
        for i in range(k):
            candidate *= primes[i]
        if candidate > 10**18:
            break
        score = _tau_perp(candidate)
        if score > best_score:
            best_score = score
            best_n = candidate
        if time.time() > deadline - 5:
            break
    
    # Now try to optimize: maybe using slightly larger primes but different combinations
    # helps. The key is that coprime adjacency depends on the structure.
    # Let's try random samples of prime subsets of size 14, 15
    
    import random
    random.seed(42)
    
    for k in [14, 15]:
        attempts = 0
        while time.time() < deadline - 10 and attempts < 2000:
            # Pick k primes from first 30
            if k <= len(primes):
                subset = random.sample(primes[:30], k)
                candidate = 1
                valid = True
                for p in subset:
                    candidate *= p
                    if candidate > 10**18:
                        valid = False
                        break
                if not valid:
                    attempts += 1
                    continue
                score = _tau_perp(candidate)
                if score > best_score:
                    best_score = score
                    best_n = candidate
            attempts += 1
    
    # Also try numbers with some exponents > 1
    # e.g., 2^a * 3 * 5 * ... where a is small
    for k in range(12, 15):
        if time.time() > deadline - 5:
            break
        # Use k-1 smallest primes, and give the smallest prime a higher exponent
        base_primes = primes[1:k]  # skip 2, use next k-1 primes
        for a in range(2, 10):
            candidate = (2 ** a)
            for p in base_primes:
                candidate *= p
            if candidate > 10**18:
                break
            score = _tau_perp(candidate)
            if score > best_score:
                best_score = score
                best_n = candidate
    
    return best_n
