import time
import math
import sys

def is_prime(n):
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    """Pollard's Brent cycle detection for finding a non-trivial factor of n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    y, c, m = 2, 1, 128
    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y + c) % n
                q = q * abs(x - y) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break
    if g == n:
        return pollard_brent(n)
    return g

def factorize(n, small_primes):
    """Fully factor n using small primes first, then Pollard-Brent."""
    factors = []
    rem = n
    # Trial division by small primes
    for p in small_primes:
        if p * p > rem:
            break
        while rem % p == 0:
            factors.append(p)
            rem //= p
    # Pollard-Brent for remaining
    if rem > 1:
        if is_prime(rem):
            factors.append(rem)
        else:
            # Recursively factor
            stack = [rem]
            while stack:
                m = stack.pop()
                if is_prime(m):
                    factors.append(m)
                else:
                    d = pollard_brent(m)
                    stack.append(d)
                    stack.append(m // d)
    return factors

def sieve(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = [True] * (limit + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, limit+1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin
    
    # Small primes for trial division
    small_primes = sieve(10000)
    
    # We'll search for primes p.
    # Strategy:
    # 1. Systematic scan of small primes (up to ~500k or as many as time allows)
    # 2. If time remains, random search of larger primes
    
    best_p = 3
    best_k = 0
    
    # Generate primes up to 500000 for systematic scan
    # pi(500000) is approx 41538
    limit_sys = 500000
    primes_sys = sieve(limit_sys)
    
    # For each prime p, check how many consecutive integers p^2+1, ..., p^2+k
    # have all prime factors <= p.
    
    def check_run(p):
        """Check the run length for prime p. Returns k."""
        base = p * p
        k = 0
        # We want to check up to 200, but can break early.
        # Also check time.
        for i in range(1, 201):
            if time.time() > deadline:
                return k
            n = base + i
            # Check if all prime factors of n are <= p
            # First, trial divide by small primes
            rem = n
            for sp in small_primes:
                if sp > p:
                    break
                if sp * sp > rem:
                    break
                while rem % sp == 0:
                    rem //= sp
            
            if rem == 1:
                k = i
                continue
            
            # Check if rem is prime and <= p
            if rem <= p:
                # rem is prime (since we divided by all primes up to sqrt(rem) or p)
                # Wait, if rem is composite, we need to factor it.
                # But if rem <= p, then any prime factor of rem is <= p.
                # So if rem is not 1, we just need to ensure rem's prime factors are <= p.
                # Since rem <= p, if rem is prime, it's <= p. If composite, its factors are smaller.
                # Actually, we just need to factor rem and check max factor <= p.
                # But since rem <= p, and we've divided by all primes <= min(p, sqrt(rem)),
                # if rem > 1, then rem is prime (or 1).
                # Wait, if rem is composite, it must have a factor <= sqrt(rem).
                # We divided by all primes <= min(p, sqrt(rem)).
                # If sqrt(rem) < p, then we divided by all primes <= sqrt(rem), so rem is prime.
                # If sqrt(rem) >= p, then we divided by all primes <= p.
                # If rem > 1 and we divided by all primes <= p, then rem's smallest prime factor > p?
                # No, we divided by primes <= p. If rem has a factor <= p, we would have divided it.
                # So if rem > 1, all prime factors of rem are > p? No, that's not right.
                # We divided by primes q <= p. If rem is divisible by q, we removed all q.
                # So if rem > 1, it is not divisible by any prime <= p.
                # Therefore, if rem > 1, all prime factors of rem are > p.
                # So rem is NOT p-smooth.
                
                # Wait, let's re-evaluate.
                # We want all prime factors of n to be <= p.
                # We divide n by small primes <= p.
                # If rem == 1, success.
                # If rem > 1, we need to check if rem is p-smooth.
                # Since we removed all prime factors <= p, if rem > 1, then rem has a prime factor > p.
                # So rem is NOT p-smooth.
                
                # Therefore, the condition is simply rem == 1.
                # Wait, what if rem is a prime > p? Then it's not p-smooth.
                # What if rem is composite with factors > p? Not p-smooth.
                # So yes, rem must be 1.
                
                pass
            
            # Actually, the logic is:
            # After dividing by all primes <= p, if rem == 1, then n is p-smooth.
            # If rem > 1, then n is NOT p-smooth.
            
            # But wait, we only divided by small_primes up to min(p, sqrt(rem)).
            # If p is large, we might not have divided by all primes <= p.
            # We need to factor rem completely and check if all factors <= p.
            
            # Let's use a more robust check:
            # Factor rem, and check max factor <= p.
            
            if rem > 1:
                # Check if rem is prime
                if rem <= p:
                    # rem is prime and <= p, so it's p-smooth
                    k = i
                    continue
                else:
                    # rem > p. Is rem prime?
                    if is_prime(rem):
                        # rem is prime > p, not p-smooth
                        break
                    else:
                        # rem is composite. Factor it and check.
                        factors = factorize(rem, small_primes)
                        if all(f <= p for f in factors):
                            k = i
                            continue
                        else:
                            break
            
            # This should be unreachable, but for safety:
            break
        
        return k
    
    # Systematic scan
    for p in primes_sys:
        if time.time() > deadline:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # If time remains, do random search of larger primes
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        tried = set(primes_sys)
        
        while time.time() < deadline - 5:
            # Pick a random number and find a nearby prime
            # We want primes up to 1e9
            # Let's pick random ranges
            low = random.randint(1000, 1000000000)
            high = low + random.randint(1000, 100000)
            # Ensure high <= 1e9
            if high > 1000000000:
                high = 1000000000
            
            # Find a prime in [low, high]
            # Start from low, make odd
            n = low if low % 2 == 1 else low + 1
            if n < 3:
                n = 3
            
            found = False
            for _ in range(100):  # Try up to 100 candidates
                if n > high:
                    break
                if is_prime(n) and n not in tried:
                    tried.add(n)
                    found = True
                    break
                n += 2
            
            if found:
                k = check_run(n)
                if k > best_k:
                    best_k = k
                    best_p = n
    
    return {'p': best_p, 'k': best_k}
