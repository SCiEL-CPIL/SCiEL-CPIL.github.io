import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        # Small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Witnesses for 64-bit
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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

    def has_factor_gt(n, p):
        """Returns True if n has a prime factor > p."""
        if n == 1:
            return False
        if n > p:
            if is_prime(n):
                return True
        # If n <= p, all factors are <= n <= p, so no factor > p
        # Wait, if n <= p, any prime factor of n is <= n <= p. So return False.
        if n <= p:
            return False
        
        # n > p and n is composite
        # Try small factors first
        # We can trial divide up to a small limit to speed up
        # But pollard rho is good.
        d = pollard_rho(n)
        if d == n:
            # Should not happen if is_prime is correct and n is composite
            # But just in case
            return True 
        # Check factor d and n//d
        if has_factor_gt(d, p):
            return True
        if has_factor_gt(n // d, p):
            return True
        return False

    def is_p_smooth(n, p):
        """Returns True if all prime factors of n are <= p."""
        return not has_factor_gt(n, p)

    # Sieve for small primes to speed up trial division and candidate generation
    sieve_limit = 10**6
    is_composite = [False] * (sieve_limit + 1)
    small_primes = []
    for i in range(2, sieve_limit + 1):
        if not is_composite[i]:
            small_primes.append(i)
            if i * i <= sieve_limit:
                for j in range(i * i, sieve_limit + 1, i):
                    is_composite[j] = True

    # Optimization: Trial divide by small primes in has_factor_gt?
    # Let's add a trial division step for small primes in is_p_smooth to reduce pollard_rho calls.
    # But has_factor_gt is recursive.
    # Let's modify has_factor_gt to try small primes first.
    
    def has_factor_gt_opt(n, p):
        if n <= p:
            return False
        if is_prime(n):
            return True
        # Trial divide by small primes
        for q in small_primes:
            if q > p: break
            if n % q == 0:
                # q is a factor. q <= p.
                # Check cofactor
                return has_factor_gt_opt(n // q, p)
        # No small factor found
        d = pollard_rho(n)
        if d == n:
            return True
        return has_factor_gt_opt(d, p) or has_factor_gt_opt(n // d, p)

    def is_p_smooth_opt(n, p):
        return not has_factor_gt_opt(n, p)

    best_k = 0
    best_p = 2
    
    # Search primes
    # We iterate through small_primes up to sieve_limit
    # Then we might check larger primes if time permits
    
    for p in small_primes:
        if time.time() > deadline:
            break
            
        # Check if p is a valid candidate (it is prime by construction)
        # Check k = 1, 2, ...
        # p^2 + 1, ...
        base = p * p
        k = 0
        while True:
            if time.time() > deadline:
                break
            val = base + k + 1
            if is_p_smooth_opt(val, p):
                k += 1
                if k > best_k:
                    best_k = k
                    best_p = p
                    # Early exit if k is very large? 
                    # The max k is 200.
                    if k >= 200:
                        # We can't get higher than 200 per problem statement? 
                        # "1 <= k <= 200". 
                        # If we hit 200, we can stop checking this p.
                        break
            else:
                break
        # If k reached 200, we can break the outer loop? 
        # No, another p might also have 200, but we want to maximize k. 
        # If max is 200, any p with 200 is equally good? 
        # The score is value/seed. Value is k. 
        # So if we find 200, we can stop entirely? 
        # Yes, 200 is the max allowed.
        if best_k >= 200:
            break

    # If time remains, search larger primes?
    # Generating primes up to 10^9 is not feasible with a simple sieve.
    # We can use Miller-Rabin to test random candidates or increment by 2/6.
    # However, if the optimal p is large, we might miss it.
    # But small primes are more likely to have smooth neighbors? 
    # Actually, the density of p-smooth numbers near p^2 is related to the Dickman function.
    # For p ~ 10^6, p^2 ~ 10^12.
    # Let's assume the best p is within the sieve range or that we can find a good one quickly.
    
    return {'p': best_p, 'k': best_k}
