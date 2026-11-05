"""Fresh design: use product-of-primes + GCD test to check p-smoothness of p^2+1..p^2+k."""
import time
import math

_DEADLINE = None

def _now():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0  # safety margin under 150s
    return time.time()

def _time_left():
    return _DEADLINE - time.time()

def _is_prime(n):
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # deterministic bases for n < 3.3e24
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _sieve_primes(n):
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = bytearray(len(range(start, n + 1, step)))
    return [i for i in range(2, n + 1) if sieve[i]]

def _product_of_primes(primes):
    prod = 1
    for p in primes:
        prod *= p
    return prod

def construct():
    _now()
    best_k = 0
    best_p = 3

    # Phase 1: small primes via direct product + GCD
    # For small p, product of primes <= p is manageable.
    # We'll go up to a limit where the product is still efficient.
    primes_small = _sieve_primes(2000)
    
    # Precompute cumulative products
    cum_prod = [1]
    for p in primes_small:
        cum_prod.append(cum_prod[-1] * p)
    
    # Map prime index to cumulative product
    # primes_small[i] corresponds to cum_prod[i+1]
    
    for idx, p in enumerate(primes_small):
        if _time_left() < 5.0:
            break
        P = cum_prod[idx + 1]  # product of all primes <= p
        p2 = p * p
        k = 0
        # Check p^2+1, p^2+2, ...
        while k < 200:
            n = p2 + k + 1
            g = math.gcd(n, P)
            if g == n:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: medium primes (2000 < p < 200000) with incremental product
    # Use a larger sieve and build product incrementally
    if _time_left() > 10.0:
        primes_med = _sieve_primes(200000)
        # Start product from primes <= 2000
        prod = cum_prod[len(primes_small)]  # product of primes <= 2000
        
        # We need to find where primes_small ends in primes_med
        # primes_small has primes up to 2000, primes_med has up to 200000
        # Find the split point
        split = len(primes_small)
        
        for idx in range(split, len(primes_med)):
            if _time_left() < 5.0:
                break
            p = primes_med[idx]
            # Update product: multiply by p
            prod *= p
            P = prod
            p2 = p * p
            k = 0
            while k < 200:
                n = p2 + k + 1
                g = math.gcd(n, P)
                if g == n:
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
    
    # Phase 3: larger primes (200000 < p < 10^7) - sparse search
    if _time_left() > 20.0:
        # For larger p, the product becomes huge. But Python handles big ints.
        # The GCD with a huge number is still fast (Euclidean algorithm on big ints).
        # Let's search a subset of primes in larger ranges.
        # We'll use a segmented approach: check primes in ranges.
        
        # Actually, for very large p, p^2 is ~10^14, and we need primes up to p.
        # The product of all primes up to 10^7 is enormous (billions of digits).
        # This won't work. Instead, for large p, use a different strategy.
        
        # Alternative: for large p, check if p^2+i is p-smooth by checking
        # gcd with product of primes up to p, but we can't compute that product.
        
        # Better: for large p, use the fact that if p^2+i has a prime factor > p,
        # that factor is at least the smallest prime > p. So p^2+i must be divisible
        # by some prime > p. We can check this by trying to divide p^2+i by primes
        # up to sqrt(p^2+i) ≈ p, but that's the same as before.
        
        # For large p, the GCD approach with full product doesn't scale.
        # Instead, let's focus on the ranges where we can afford the product.
        
        # Let's try primes up to 10^6 with the product approach.
        if _time_left() > 15.0:
            primes_large = _sieve_primes(1000000)
            # Continue product from where we left off
            # prod currently = product of primes <= 200000
            # Need to extend to 10^6
            # Find split point in primes_large
            split2 = len(primes_med)
            
            for idx in range(split2, len(primes_large)):
                if _time_left() < 5.0:
                    break
                p = primes_large[idx]
                prod *= p
                P = prod
                p2 = p * p
                k = 0
                while k < 200:
                    n = p2 + k + 1
                    g = math.gcd(n, P)
                    if g == n:
                        k += 1
                    else:
                        break
                if k > best_k:
                    best_k = k
                    best_p = p
    
    return {'p': best_p, 'k': best_k}
