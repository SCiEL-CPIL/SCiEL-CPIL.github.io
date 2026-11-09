import time
import math

def _powerful_part(m):
    """Return the powerful part of m: product of p^e for all prime powers p^e || m with e >= 2."""
    if m <= 1:
        return 1
    result = 1
    temp = m
    # Factor out small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p * p > temp:
            break
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                result *= p ** e
    # If remaining temp > 1, it's a prime factor with exponent 1
    # (since if it were p^2 with p >= 97, p^2 >= 9409, but we'd have caught smaller factors)
    # Actually, if temp > 1 and temp is prime, exponent is 1, so we skip it.
    # If temp is a perfect square of a prime > 97, we need to check.
    if temp > 1:
        # Check if temp is a perfect square
        r = int(math.isqrt(temp))
        if r * r == temp and r > 1:
            # temp = r^2, but r might be composite. However, since we've divided out all primes <= 97,
            # if r is prime and > 97, then temp = r^2 contributes r^2.
            # If r is composite, it would have prime factors we already handled... but we divided those out.
            # So r must be prime (or 1).
            if r > 97:
                result *= temp
            # else r <= 97 but we should have caught it... unless r is prime > 97? No, r <= 97.
            # Actually if r <= 97 and r is prime, we would have divided it out. So this case shouldn't happen
            # unless r is composite, but then its prime factors are <= 97 and already removed.
            # So temp being a perfect square with r <= 97 means r's prime factors were already removed,
            # meaning temp should be 1. Contradiction. So only r > 97 matters.
    return result

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    best_n = 10000
    best_score = 0
    
    # Strategy 1: Scan all perfect squares n = a^2 in [10000, 1000000]
    # powerful_part(n) = n for perfect squares
    # Score = n * powerful_part(n+1)
    a_start = 100
    a_end = 1000
    for a in range(a_start, a_end + 1):
        if time.time() > deadline:
            break
        n = a * a
        if n < 10000 or n > 1000000:
            continue
        pp_n = n  # powerful part of a perfect square is itself
        pp_np1 = _powerful_part(n + 1)
        score = pp_n * pp_np1
        if score > best_score:
            best_score = score
            best_n = n
    
    # Strategy 2: Scan a range of n values, computing powerful_part(n) and powerful_part(n+1)
    # Focus on n where powerful_part(n) is large
    # We can do a broader scan but it's expensive. Let's do a targeted scan.
    # For n not a perfect square, powerful_part(n) < n typically.
    # Let's scan n in steps, but prioritize values where n or n+1 is divisible by high prime powers.
    
    # Additional strategy: check n where n+1 is a perfect square
    # n+1 = a^2 => n = a^2 - 1
    for a in range(101, 1001):  # a^2 in [10201, 1002001], n in [10200, 1002000]
        if time.time() > deadline:
            break
        n = a * a - 1
        if n < 10000 or n > 1000000:
            continue
        pp_n = _powerful_part(n)
        pp_np1 = a * a  # n+1 is a perfect square
        score = pp_n * pp_np1
        if score > best_score:
            best_score = score
            best_n = n
    
    # Strategy 3: Broader scan - check n values where n has a large powerful part
    # Sample n values, compute powerful_part(n) * powerful_part(n+1)
    # To be efficient, scan in chunks but check time
    # Let's do a denser scan around promising regions
    # Check all n in [10000, 1000000] step by step is too slow (990000 iterations)
    # But _powerful_part is fast for small numbers. Let's try a moderate scan.
    
    # Do a scan of all n in range, but with time check
    # 990000 iterations, each _powerful_part call is fast (small primes only)
    # Let's try it
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
        pp_n = _powerful_part(n)
        pp_np1 = _powerful_part(n + 1)
        score = pp_n * pp_np1
        if score > best_score:
            best_score = score
            best_n = n
    
    return best_n
