import time
import math
import random

# Sieve of Eratosthenes to get primes up to 10^6
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

_SMALL_PRIMES = sieve(1000000)
_SMALL_PRIME_SET = set(_SMALL_PRIMES)

# Deterministic Miller-Rabin for 64-bit integers
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
        m = 128
        g = r = q = 1
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
        if g != n:
            return g

def is_p_smooth(n, p):
    """
    Check if all prime factors of n are <= p.
    """
    if n <= 1:
        return True
    
    # Trial division by small primes up to min(p, 10^5)
    limit = min(p, 100000)
    for q in _SMALL_PRIMES:
        if q > limit:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
    
    if n == 1:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
    
    # Otherwise, n is composite.
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            if x > p:
                return False
            continue
        
        r = int(math.isqrt(x))
        if r * r == x:
            stack.append(r)
            continue
            
        d = pollard_brent(x)
        stack.append(d)
        stack.append(x // d)
        
    return True

def check_range_smooth(p, k):
    """
    Check if all integers p^2+1, ..., p^2+k have all prime factors <= p.
    Uses segmented sieve for fast batch verification.
    Returns the maximum valid k' <= k, or 0 if p^2+1 is not p-smooth.
    """
    if k <= 0:
        return 0
    
    base = p * p
    # We need to check numbers base+1 through base+k
    
    # For each number in the range, we need to verify all prime factors <= p.
    # Strategy: 
    # 1. Trial divide by all primes q <= min(p, sqrt(base+k))
    # 2. After trial division, remaining factor must be 1 or a prime <= p
    
    # Generate primes up to min(p, sqrt(base+k))
    # base+k is at most p^2 + 200, so sqrt(base+k) is at most p + small
    # Actually sqrt(p^2 + 200) < p + 1 for p >= 15
    # So we need primes up to p for trial division? That's too many for large p.
    
    # Better approach: 
    # For each prime q <= p, divide out q from all numbers in the range.
    # After processing all q <= p, if any number has remaining factor > 1, 
    # that factor must be prime (since all its prime factors are > p, but we've 
    # divided out all primes <= p). Wait, that's not right.
    
    # Let me think again. We want to check if n has all prime factors <= p.
    # Equivalent: after dividing out all prime powers q^a where q <= p, 
    # the remaining value must be 1.
    
    # For a range of k consecutive numbers, we can do segmented sieve:
    # For each prime q <= p, mark all multiples of q in the range and divide them out.
    
    # But p can be up to 10^9, so we can't iterate over all primes <= p.
    
    # Key insight: For n = p^2 + i (1 <= i <= k), any prime factor q of n satisfies:
    # - If q <= p: we can divide it out
    # - If q > p: then n is NOT p-smooth
    # 
    # So n is p-smooth iff after removing all prime factors <= p, the remainder is 1.
    # Equivalently, n has no prime factor > p.
    #
    # Since n <= p^2 + 200 < (p+1)^2 for p > 200, n has at most 2 prime factors 
    # (counting multiplicity) if both are > 1... wait no.
    #
    # Actually, if n has a prime factor q > p, then n/q < (p^2+200)/p = p + 200/p < p + 1.
    # So n/q < p+1, meaning n/q <= p (for integer).
    # 
    # So if n has a prime factor > p, then n = q * m where q > p and m <= p.
    # This means n is composite and has exactly one prime factor > p (the largest one),
    # and the cofactor m <= p.
    #
    # Therefore, to check if n is p-smooth:
    # 1. Check if n is prime: if yes, n <= p? But n = p^2+i > p for i >= 1, so n is not p-smooth if prime.
    #    Wait, n = p^2 + 1 > p always, so if n is prime, it's NOT p-smooth.
    # 2. If n is composite, check if it has a prime factor > p.
    #    As argued, if it does, then n = q*m with q > p prime and m <= p.
    #    So we can check: does n have a divisor q such that q > p and q is prime?
    #    Equivalently: does n have a divisor m <= p such that n/m > p and n/m is prime?
    #    Or: for each divisor m of n with m <= p, check if n/m is prime and > p.
    
    # This is still hard for large n.
    
    # Alternative: use the fact that n = p^2 + i.
    # Any prime factor q of n with q > p must satisfy q | (p^2 + i).
    # Since q > p, q does not divide p, so p has an inverse mod q.
    # p^2 ≡ -i (mod q), so p ≡ ±sqrt(-i) (mod q).
    # This means q divides some value related to i.
    
    # Actually, let's just use a different approach:
    # For each i in 1..k, compute n = p^2 + i.
    # Trial divide n by all primes q <= min(p, n^(1/2)).
    # If after trial division, the remaining factor r > 1:
    #   - If r is prime and r <= p: smooth
    #   - If r is prime and r > p: not smooth
    #   - If r is composite: need to check further, but r < n and we've removed small factors.
    
    # The key optimization: for the range p^2+1 to p^2+k, we can batch the trial division.
    
    # Let's implement a segmented approach:
    # Create an array rem[1..k] where rem[i] = p^2 + i
    # For each prime q <= p:
    #   For each multiple of q in the range [p^2+1, p^2+k], divide rem[i] by q as many times as possible.
    # After processing all q <= p, check if all rem[i] == 1.
    
    # But we can't iterate over all primes q <= p if p is large.
    
    # However, we only need to process primes q where q has a multiple in [p^2+1, p^2+k].
    # For prime q, the multiples in the range are:
    #   first = ceil((p^2+1)/q) * q
    #   last = floor((p^2+k)/q) * q
    #   if first <= last, then there are multiples.
    
    # Number of multiples ≈ k/q.
    # Total work over all q: sum_{q <= p} k/q ≈ k * log(log(p))
    # This is O(k log log p) which is very fast!
    
    # But we need to iterate over primes q <= p. For p up to 10^9, that's ~50 million primes.
    # However, we only need to consider primes q where there's actually a multiple in the range.
    # For q > k, there's at most 1 multiple. For q > p^2 + k, there are 0 multiples.
    # Actually, for q > p^2 + k, no multiples. For p^2/k < q <= p^2 + k, at most 1 multiple.
    # For k < q <= p^2/k, at least 1 multiple.
    
    # Hmm, this is getting complex. Let me just iterate over primes q up to min(p, p^2+k) 
    # but only those that have a multiple in the range.
    
    # Actually, simpler: just iterate over primes q <= p. For each, check if there's a multiple.
    # The number of primes <= p is ~p/ln(p). For p = 10^6, that's ~72000. For p = 10^7, ~664000.
    # For each prime, we do O(k/q) work. Total: O(k * sum_{q<=p} 1/q) = O(k log log p).
    
    # But iterating over 72000 primes for each p candidate, and we might check many p's...
    # This should be fast enough.
    
    rem = [base + i for i in range(1, k+1)]
    
    # Get primes up to min(p, base+k) -- but we only need primes <= p
    # Use our precomputed small primes for q <= 10^6
    # For q > 10^6 and q <= p, we need to generate them
    
    # Actually, let's just use the precomputed primes up to 10^6 first, 
    # then handle larger primes if needed.
    
    # For q > 10^6 and q <= p:
    #   The number of multiples in [base+1, base+k] is either 0 or 1 (since k=200 < q)
    #   So we can check: is there a multiple of q in the range?
    #   Multiple = m*q where m = ceil((base+1)/q)
    #   If m*q <= base+k, then that multiple is in range.
    #   But we need to know if q is prime.
    
    # This is getting complicated. Let me just use the precomputed primes up to 10^6,
    # and for the remaining factor, use Miller-Rabin + Pollard-Brent as fallback.
    
    # Phase 1: Trial divide by precomputed primes q <= min(p, 10^6)
    for q in _SMALL_PRIMES:
        if q > p:
            break
        # Find multiples of q in [base+1, base+k]
        first = ((base + 1 + q - 1) // q) * q
        if first > base + k:
            continue
        for m in range(first, base + k + 1, q):
            idx = m - base - 1  # 0-indexed
            while rem[idx] % q == 0:
                rem[idx] //= q
    
    # Phase 2: For each remaining rem[i] > 1, check if it's p-smooth
    # After phase 1, rem[i] has no prime factors <= min(p, 10^6)
    # If rem[i] == 1: smooth
    # If rem[i] > 1: 
    #   - If rem[i] is prime: smooth iff rem[i] <= p
    #   - If rem[i] is composite: all its prime factors are > 10^6 (or > p if p < 10^6)
    #     Need to check if all factors <= p
    
    for i in range(k):
        if rem[i] == 1:
            continue
        # Check if rem[i] is p-smooth
        if not is_p_smooth(rem[i], p):
            return i  # The first i where it's not smooth (0-indexed), so valid k = i
        # rem[i] is smooth, continue
    
    return k

def construct():
    deadline = time.time() + 115
    best_k = 0
    best_p = 0
    
    # Try primes in increasing order
    for p in _SMALL_PRIMES:
        if time.time() > deadline:
            break
        if p > 10**9:
            break
        
        # Quick check: is p^2+1 p-smooth?
        # If not, skip this p entirely
        k = check_range_smooth(p, 200)
        
        if k > best_k:
            best_k = k
            best_p = p
        
        # Early termination: if we've found k=200, that's the max possible
        if best_k >= 200:
            break
    
    # If we have time, try some larger primes
    if time.time() < deadline - 5:
        last_p = _SMALL_PRIMES[-1]
        while time.time() < deadline - 2:
            last_p += 2
            if last_p > 10**9:
                break
            if not is_prime(last_p):
                continue
            
            p = last_p
            k = check_range_smooth(p, 200)
            
            if k > best_k:
                best_k = k
                best_p = p
            
            if best_k >= 200:
                break
    
    if best_k < 1:
        # Fallback: find any valid p,k
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}
