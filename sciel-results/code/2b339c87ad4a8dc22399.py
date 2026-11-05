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
    #    a. If n <= p^2, we can check directly.
    #    b. We divide out all small primes <= p.
    #    c. If the remainder is 1, it's p-smooth.
    #    d. If the remainder is prime and <= p, it's p-smooth.
    #    e. If the remainder is composite, it must have a factor > p (since we divided out all <= p). 
    #       Wait, if we divide out all primes <= p, the remainder is 1 or a product of primes > p.
    #       So if remainder > 1, it has a prime factor > p.
    #    Optimization: We only need to divide by primes <= p.
    #    If p is large (e.g., 10^8), we can't iterate all primes up to p.
    #    However, n is roughly p^2. 
    #    If n is p-smooth, all its factors are <= p.
    #    Let n = p1 * p2 * ... * pk.
    #    The largest prime factor must be <= p.
    #    We can try to factor n. If we find a prime factor > p, return False.
    #    
    #    Revised Strategy for is_p_smooth(n, p):
    #    1. If n <= p: return True (since n < p^2 usually, but if n=p, factor is p)
    #       Actually, if n <= p, largest factor is n <= p. True.
    #    2. We need to ensure no prime factor > p.
    #    3. This is equivalent to: Let m = n.
    #       For each prime q <= sqrt(m):
    #           if q > p: break? No, if q > p, then q is a factor > p. Return False.
    #           while m % q == 0: m //= q
    #       After loop, if m > 1:
    #           if m <= p: return True
    #           else: return False (since m is prime or 1, and if > p, it's a factor > p)
    #    
    #    Wait, the loop "For each prime q <= sqrt(m)" requires iterating primes.
    #    If p is large, sqrt(n) ~ p. We would iterate up to p.
    #    But we only have small_primes up to 10^6.
    #    If p > 10^6, we need primes up to p.
    #    
    #    Alternative:
    #    If p is small, we have all primes up to p. We divide them out.
    #    If p is large, we only have primes up to 10^6.
    #    Let R be the remainder after dividing by all small primes <= min(p, 10^6).
    #    If R == 1: True.
    #    If R is prime:
    #        if R <= p: True
    #        else: False
    #    If R is composite:
    #        R = A * B. Since we removed all factors <= 10^6, A and B > 10^6.
    #        So R > 10^12.
    #        But n ~ p^2. If p > 10^6, n > 10^12.
    #        If R is composite and > 1, it has a prime factor.
    #        Actually, if we removed all factors <= 10^6, and R is composite, its smallest prime factor is > 10^6.
    #        We need to check if R has any prime factor <= p.
    #        If p > 10^6, we don't know if R's factors are <= p.
    #        
    #    Let's simplify:
    #    The condition "all prime factors <= p" is equivalent to:
    #    Largest prime factor of n <= p.
    #    
    #    We can compute the largest prime factor of n.
    #    If n is not too large, we can factor it.
    #    For n ~ 10^18 (p=10^9), factoring is hard.
    #    But we don't need full factorization, just to check if LPF(n) <= p.
    #    
    #    If we divide n by all primes <= p, and get 1, then LPF(n) <= p.
    #    Problem: Iterating primes up to p is too slow for p=10^9.
    #    
    #    Insight:
    #    If n has a prime factor > p, then n = k * q where q > p is prime.
    #    Then k = n/q < n/p ~ p.
    #    So if we find a divisor k < p, we can check if n/k is prime and > p.
    #    
    #    Actually, simpler:
    #    If LPF(n) > p, then n has a prime factor q > p.
    #    Since q > p, and n < p^2 + k (small k), n < p^2 + 200.
    #    So q < p^2 + 200.
    #    Also k = n/q < (p^2+200)/p = p + 200/p < p + 1.
    #    So k < p.
    #    Thus, if n has a prime factor > p, it must have a co-factor k < p.
    #    And q = n/k must be prime.
    #    
    #    So, to check if n is p-smooth:
    #    1. Check if n has any prime factor q > p.
    #    2. This is equivalent to: Does there exist a divisor k of n such that k < p and n/k is prime and n/k > p?
    #    
    #    How to check this fast?
    #    We can iterate possible k? No, k can be up to p.
    #    
    #    Let's go back to division.
    #    Divide n by all primes <= 10^6. Let R be the remainder.
    #    If R == 1: True.
    #    If R is prime:
    #        If R <= p: True
    #        Else: False
    #    If R is composite:
    #        R must be a product of primes > 10^6.
    #        Let R = a * b. a, b > 10^6.
    #        Since n < p^2 + 200, R < p^2 + 200.
    #        If p is large (e.g., > 10^6), R could be composite.
    #        But if R is composite, its smallest prime factor a > 10^6.
    #        We need to check if a <= p.
    #        If a <= p, then b = R/a. Is b <= p?
    #        If b <= p, then all factors are <= p? 
    #        Wait, if R is composite, it has at least two prime factors.
    #        If both are <= p, then it's p-smooth.
    #        If one is > p, it's not.
    #        
    #    Since we only have primes up to 10^6, we can't easily factor R if R is composite and large.
    #    However, if R is composite, it must have a factor <= sqrt(R).
    #    sqrt(R) <= sqrt(n) ~ p.
    #    If p <= 10^6, we have all primes up to p, so we can fully factor n.
    #    If p > 10^6, we don't have primes up to p.
    #    
    #    But note: if p > 10^6, then n ~ p^2 > 10^12.
    #    If n has a factor > p, then n = k * q with q > p, k < p.
    #    k < p.
    #    If k has a prime factor <= 10^6, we would have removed it?
    #    No, we removed factors from n.
    #    
    #    Let's try a different approach for p > 10^6:
    #    We know n = p^2 + i.
    #    If n is not p-smooth, it has a prime factor q > p.
    #    Then n = k * q.
    #    q > p => k < n/p ~ p.
    #    So k is a divisor of n, k < p.
    #    Also q = n/k is prime.
    #    
    #    We can check if n has any prime factor > p by checking if n has any divisor k < p such that n/k is prime and > p.
    #    But finding divisors is hard.
    #    
    #    Alternative:
    #    If p is large, the probability that n is p-smooth is low?
    #    Actually, for random n ~ p^2, the probability that all factors are <= p is related to the distribution of prime factors.
    #    
    #    Let's stick to the small prime division.
    #    If p <= 10^6, we can fully factor n using small_primes.
    #    If p > 10^6, we divide by small_primes up to 10^6.
    #    If R == 1: True.
    #    If R is prime:
    #        If R <= p: True
    #        Else: False
    #    If R is composite:
    #        We need to check if R has any prime factor > p.
    #        Since R is composite and all its factors are > 10^6, 
    #        and R < p^2 + 200.
    #        If R has a factor > p, then R = a * b with a > p.
    #        Then b < R/p < p.
    #        So b < p.
    #        But b is a factor of R.
    #        We don't have primes up to b if b > 10^6.
    #        However, if b < p and p > 10^6, b could be large.
    #        
    #    Wait, if R is composite, let a be the smallest prime factor of R.
    #    a > 10^6.
    #    If a > p, then R is not p-smooth.
    #    If a <= p, then R/a is an integer.
    #    Let R' = R/a.
    #    If R' == 1: True (since a <= p).
    #    If R' > 1:
    #        Check if R' has any prime factor > p.
    #        This is recursive.
    #        
    #    Given the time constraint, maybe we should restrict search to p <= 10^6?
    #    The current seed program searches p < 2000.
    #    If we search up to 10^6, we can use the full small_primes list.
    #    This is a significant jump in search space.
    
    best_k = 0
    best_p = 3

    # Search primes up to 10^6
    # We can generate primes on the fly or use the sieve.
    # We have small_primes up to 10^6.
    
    for p in small_primes:
        if time.time() > deadline:
            break
            
        # Skip small p if we already have a good k?
        # For now, just check all.
        
        # Check p^2 + 1 ... p^2 + k
        k = 0
        base = p * p
        
        # We want to maximize k.
        # Check consecutive integers.
        
        # Optimization: If p is small, we can check many.
        # If p is large, checking is slower.
        
        # Let's try to find the max k for this p.
        # We stop when we find a number that is NOT p-smooth.
        
        # To check if n is p-smooth for p <= 10^6:
        # We have all primes up to p.
        # We divide n by all primes <= p.
        # If remainder is 1, it's p-smooth.
        
        # But doing this for every n is slow.
        # We can precompute primes up to p? 
        # small_primes is sorted.
        # We can binary search the index of p in small_primes.
        
        # For a given p, let primes_p = small_primes[:idx_p].
        # For each n = p^2 + i:
        #   temp = n
        #   for q in primes_p:
        #       if q*q > temp: break
        #       while temp % q == 0:
        #           temp //= q
        #   if temp == 1 or (is_prime_mr(temp) and temp <= p):
        #       k += 1
        #   else:
        #       break
        
        # This is O(k * pi(p) * log(n)).
        # For p=10^6, pi(p) ~ 78,000.
        # k is likely small.
        # But checking each n takes ~78,000 divisions?
        # No, we break early if q*q > temp.
        # For n ~ 10^12, sqrt(n) ~ 10^6.
        # So we might iterate all primes up to p.
        # This is too slow for many p's.
        
        # Better:
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
        
        idx_p = 0
        # Find index of p in small_primes
        # Since p is from small_primes, we can just use the index in the loop.
        
        # Let's iterate with index.
        # small_primes is a list.
        
    # Re-iterate properly
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
