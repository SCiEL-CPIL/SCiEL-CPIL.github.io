import math
import time
import random

def construct():
    deadline = time.time() + 110.0
    
    # 1. Sieve primes up to 10^6
    SIEVE_LIMIT = 10**6
    is_prime = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime[0:2] = b'\x00\x00'
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime[i]]
    prime_set = set(primes)

    def is_prime_mr(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        # Check small primes first
        for sp in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == sp: return True
            if n % sp == 0: return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        y, c, m = random.randint(1, n-1), random.randint(1, n-1), 128
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r-k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1: break
        return g

    def is_p_smooth(n, p, primes_list):
        """
        Check if n is p-smooth.
        primes_list contains all primes <= 10^6.
        If p <= 10^6, we use primes_list up to p.
        If p > 10^6, we use all primes in primes_list, then handle the remainder.
        """
        if n <= 1: return True
        
        # Trial divide by primes <= min(p, 10^6)
        # Since primes_list is sorted, we can iterate until prime > p or prime > 10^6
        limit = p if p < SIEVE_LIMIT else SIEVE_LIMIT
        for q in primes_list:
            if q > limit:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        
        if n == 1:
            return True
            
        # If p <= 10^6, we have divided by all primes <= p.
        # If n > 1, it has a prime factor > p.
        if p <= SIEVE_LIMIT:
            return False
        
        # If p > 10^6, we have divided by all primes <= 10^6.
        # n is now either 1, prime, or composite with factors > 10^6.
        # We need to check if all prime factors of n are <= p.
        # Since we removed factors <= 10^6, any remaining factor is > 10^6.
        # If n is prime, check n <= p.
        # If n is composite, it has factors > 10^6. 
        # We can use Pollard's Rho to factor n completely and check max factor.
        
        # Optimization: If n is small enough, trial division by primes > 10^6 is hard.
        # Use Pollard's Rho to find a factor.
        
        factors = []
        stack = [n]
        while stack:
            curr = stack.pop()
            if curr == 1: continue
            if is_prime_mr(curr):
                factors.append(curr)
                continue
            # Try to find a factor
            f = pollard_brent(curr)
            if f == curr:
                # Failed to find factor, treat as prime (should be rare)
                factors.append(curr)
            else:
                stack.append(f)
                stack.append(curr // f)
        
        # Check if all factors are <= p
        for f in factors:
            if f > p:
                return False
        return True

    def check_run(p, primes_list):
        p_sq = p * p
        k = 0
        for j in range(1, 201):
            n = p_sq + j
            if is_p_smooth(n, p, primes_list):
                k = j
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # 2. Exhaustive search for p <= 10^6
    # This is fast because is_p_smooth uses only trial division for p <= 10^6
    for p in primes:
        if time.time() > deadline:
            break
        k = check_run(p, primes)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 100: # Unlikely, but stop if very high
                break

    # 3. Search for larger primes
    # Random search in [10^6, 10^9]
    # We use a hybrid is_p_smooth which uses Pollard's Rho for the remainder
    random.seed(42)
    checked = set()
    
    # We can also try primes just above 10^6
    # Let's generate some candidates
    # Or just random
    
    while time.time() < deadline:
        # Pick a random number in [10^6, 10^9]
        # Bias towards smaller? 
        x = random.randint(10**6, 10**9)
        if x in checked:
            continue
        checked.add(x)
        
        # Quick primality check
        if not is_prime_mr(x):
            continue
        
        p = x
        k = check_run(p, primes)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 100:
                break

    return {'p': best_p, 'k': best_k}
