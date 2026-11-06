import math
import random
import time

def is_prime(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic Miller-Rabin for n < 3,215,031,751
    # Witnesses: 2, 3, 5, 7, 11
    # For n < 10^9, 2, 3, 5, 7 are enough?
    # Actually, for n < 2,152,302,898,747, witnesses 2,3,5,7,11 are enough.
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factorize(n):
    factors = []
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            factors.append(m)
        else:
            d = pollard_rho(m)
            stack.append(d)
            stack.append(m // d)
    return factors

def largest_prime_factor(n):
    if n == 1:
        return 1
    factors = factorize(n)
    return max(factors)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_p = 7
    best_k = 1
    
    # Generate primes up to some limit
    # We don't know how far to search. Let's search primes in increasing order.
    # If we find a k > best_k, update.
    # We can also try random large primes if time permits.
    
    # First, check small primes to establish a baseline.
    # We know p=41 gives k=2.
    
    # Let's iterate through primes.
    # Sieve for primes up to 10^6 first.
    limit = 10**6
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    for p in primes:
        if time.time() > deadline:
            break
        if p < 3:
            continue
        k = 0
        # Check j=1, 2, ...
        # We can stop if k exceeds best_k and we just want to confirm?
        # No, we want to maximize k.
        # If k reaches 200, we can stop checking this p.
        while k < 200:
            n = p * p + k + 1
            # Check if n is p-smooth
            # We can optimize: if n has a factor > p, it fails.
            # Trial division by primes up to p is too slow for large p.
            # Use factorize.
            lp = largest_prime_factor(n)
            if lp <= p:
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a very good k, we might want to search nearby?
            # Or just continue.
            # If best_k is high, we can stop early? No, keep searching.
    
    # If time remains, search larger primes randomly or sequentially
    # But sequential primes > 10^6 require primality testing.
    # Let's check if we have time.
    if time.time() < deadline - 10:
        # Check some larger primes
        # We can sample primes in ranges
        # For simplicity, let's just iterate from last checked prime upwards, but slowly
        # Or just return the best found so far.
        # Given the probability, finding k > 2 is rare.
        # k=3 requires 3 consecutive p-smooth numbers.
        # Let's try to search a bit more if we found k=2.
        
        # Let's try to search for primes that are congruent to something?
        # No obvious pattern.
        
        # Let's just return the best found.
        pass

    return {'p': best_p, 'k': best_k}
