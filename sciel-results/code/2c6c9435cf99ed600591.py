import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds deadline
    
    # Sieve of Eratosthenes for small primes
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    small_primes = sieve(100000) # Primes up to 10^5 for trial division
    
    # Miller-Rabin primality test
    def is_prime_mr(n):
        if n < 2:
            return False
        # Small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d*2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Bases sufficient for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho (Brent variant)
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
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
                        q = (q * abs(x - y)) % n
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

    def factorize(n):
        if n == 1:
            return []
        if n in small_primes_set:
            return [n]
        factors = []
        # Trial division by small primes
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Use Pollard-Brent
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    small_primes_set = set(small_primes)

    def largest_prime_factor(n):
        if n == 1:
            return 1
        # Quick check for small factors
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                if n == 1:
                    return p
                # If remaining n is prime, it's the largest
                if is_prime_mr(n):
                    return max(p, n)
                # Otherwise, we need full factorization of n
                # But we can just call factorize on the remainder
                break
        
        # If we broke out of loop or n was not divisible
        # We need to find the largest factor of the original n
        # Actually, let's just use factorize for simplicity if not trivial
        # But factorize is slow. Let's optimize.
        
        # Re-do:
        # We want max factor.
        # Trial divide by small primes.
        # Keep track of max factor found so far.
        max_f = 1
        temp_n = n
        for p in small_primes:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                max_f = max(max_f, p)
                while temp_n % p == 0:
                    temp_n //= p
        if temp_n > 1:
            if is_prime_mr(temp_n):
                max_f = max(max_f, temp_n)
            else:
                # temp_n is composite. Find its largest prime factor.
                # Use factorize
                f = factorize(temp_n)
                max_f = max(max_f, max(f))
        return max_f

    best_k = 0
    best_p = 2
    
    # Generate primes for p
    # We can scan primes up to 10^7
    # Let's generate primes in batches or just up to 10^7
    limit_p = 10**7
    primes_p = sieve(limit_p)
    
    for p in primes_p:
        if time.time() > deadline:
            break
        
        k = 0
        # Check p^2 + 1, ..., p^2 + 200
        # We can stop early if k < best_k and we've already checked enough?
        # No, we need to find the run length.
        # But if we already have best_k = 10, and we find k=2 for this p, we don't need to check further?
        # Wait, the run must be contiguous from p^2+1.
        # So if p^2+1 is bad, k=0.
        # If p^2+1 is good, p^2+2 is bad, k=1.
        # So we just check sequentially.
        
        # Optimization: If best_k is large, and we find a break at j < best_k, 
        # we can stop checking this p? No, because we need to know if it breaks later?
        # No, the run length is the FIRST break.
        # So if it breaks at j=5, k=5.
        # If k=5 and best_k=10, we don't care.
        # So we can stop checking this p if j > best_k? 
        # No, if j=1 is bad, k=0.
        # If j=1..best_k are good, but j=best_k+1 is bad, k=best_k.
        # So we only need to check up to best_k+1 to see if we can beat best_k.
        # If we pass best_k+1, we update best_k.
        
        p_sq = p * p
        # Check j from 1 to min(200, best_k + 1)
        # If we pass best_k+1, we extend best_k and continue.
        
        j = 1
        while j <= 200:
            n = p_sq + j
            # Check if largest factor <= p
            if largest_prime_factor(n) > p:
                break
            if j > best_k:
                best_k = j
                best_p = p
            j += 1
            
            # Optimization: if j > best_k + 10 and time is running out?
            # No, just keep going.
            
    return {'p': best_p, 'k': best_k}
