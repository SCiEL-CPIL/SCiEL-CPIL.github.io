import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit

    # 1. Generate a list of small primes up to 10^6 for fast trial division
    limit_small = 10**6
    sieve = [True] * (limit_small + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # 2. Helper: Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        # Check small primes
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                return n == p
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Bases sufficient for n < 2^64
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

    # 3. Helper: Check if n has all prime factors <= p
    #    Strategy: 
    #    a. Divide out all small primes <= min(p, 10^6).
    #    b. If remainder is 1, it's p-smooth.
    #    c. If remainder is prime and <= p, it's p-smooth.
    #    d. If remainder is composite, we need to factor it further if possible.
    #       But if p <= 10^6, we have all primes up to p, so remainder must be 1 or prime <= p.
    #       If p > 10^6, remainder could be composite with factors > 10^6.
    #       In that case, if remainder > 1, we check if it's prime.
    #       If prime and <= p: True.
    #       If prime and > p: False.
    #       If composite: It has a factor > 10^6. 
    #       Since n ~ p^2, and we removed factors <= 10^6, the remainder R < p^2.
    #       If R is composite, it has a factor a <= sqrt(R) < p.
    #       But we don't have primes up to p if p > 10^6.
    #       However, if p > 10^6, the probability of such a composite remainder being p-smooth is low?
    #       Actually, if R is composite, it has a prime factor q.
    #       If q > p, then False.
    #       If q <= p, then R/q is an integer.
    #       This is hard to check without primes up to p.
    #       
    #    Given the time constraint, we will restrict our search to p <= 10^6 
    #    where we can fully verify p-smoothness using small_primes.
    #    This is a significant jump from the previous p < 2000.
    
    best_k = 0
    best_p = 3

    # Search primes up to 10^6
    # We can generate primes on the fly or use the sieve.
    # We have small_primes up to 10^6.
    
    # Optimization: Precompute indices for small_primes?
    # small_primes is sorted.
    
    # To speed up the inner loop, we can avoid iterating all primes up to p if p is large.
    # But for p <= 10^6, pi(p) is at most 78,498.
    # For each n, we iterate primes until q*q > temp.
    # For n ~ 10^12, sqrt(n) ~ 10^6. So we might iterate many primes.
    # This is slow.
    
    # Alternative:
    # We only need to check if n has a factor > p.
    # If we divide by all primes <= p, and temp > 1, then temp has a factor > p.
    # So we just need to check if temp == 1.
    
    # But dividing by 78,000 primes for each n is slow.
    
    # Let's limit the search to p < 10^5 first to be safe, then expand if time permits.
    # Or just run it and see.
    
    # Actually, we can optimize the division loop.
    # If p is large, n is large.
    # Most n will have a small prime factor.
    # So the loop will break early.
    
    # Let's iterate with index.
    # small_primes is a list.
    
    for i, p in enumerate(small_primes):
        if time.time() > deadline:
            break
            
        # Optional: skip if p is very small and we already have high k?
        # No, small p might have long runs?
        # Actually, for small p, p^2 is small, factors are small.
        # But p is the limit.
        
        k = 0
        base = p * p
        
        # Check n = base + 1, base + 2, ...
        
        # We can break early if k is already less than best_k and p is large?
        # No, k can grow.
        
        # Let's just check.
        
        # To speed up, we can check a few first.
        
        for offset in range(1, 201): # Max k is 200
            if time.time() > deadline:
                break
                
            n = base + offset
            temp = n
            
            # Divide by primes <= p
            # Use primes from small_primes[0:i+1]
            
            # Optimization: 
            # If temp becomes 1, break.
            # If q*q > temp, break.
            
            # Iterate primes up to p
            # We can use a local list for speed?
            # primes_p = small_primes[:i+1] is a copy, slow.
            # Just iterate small_primes and break when prime > p.
            
            for q in small_primes:
                if q > p:
                    break
                if q * q > temp:
                    break
                while temp % q == 0:
                    temp //= q
            
            if temp == 1:
                k += 1
            else:
                # temp > 1
                # Check if temp is prime and <= p
                # If temp <= p, it must be prime (since we divided out all smaller primes)
                # If temp > p, it's not p-smooth.
                if temp <= p:
                    k += 1
                else:
                    break
            
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
