import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n == 2 or n == 3
        # Deterministic Miller-Rabin for 64-bit
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Witnesses sufficient for n < 3.4 * 10^14
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
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



    # Pre-compute small primes for trial division speedup in the main loop
    # We'll use a simple sieve up to 100000 for initial filtering
    limit = 100000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Helper to check if n has any prime factor > p
    # We divide by small primes first. If the remainder is 1, it's good.
    # If the remainder is prime and <= p, it's good.
    # If the remainder is composite, we need to factor it to be sure.
    # But if we just want to know if there's a factor > p, and we've divided all small primes,
    # the remaining factor is either 1, a prime > p, or a product of primes > p.
    # Wait, if we divide by all primes <= p, the remainder is 1.
    # But p can be huge. We can't divide by all primes <= p.
    # We can only divide by small primes.
    # If after dividing by small primes, the remainder R is 1, then all factors are small (and thus <= p, assuming p is large enough).
    # If R > 1, we must check if R has a factor > p.
    # Since we only divided by small primes, R is composed of primes > 10000.
    # If R is prime, it's a factor. If R > p, then R > p is a factor? No, R is a factor.
    # If R is prime and R > p, then we have a factor > p.
    # If R is composite, it has factors > 10000.
    # If p < 10000, we should have caught small factors.
    # Strategy:
    # 1. Divide n by all small_primes.
    # 2. Let R be the remainder.
    # 3. If R == 1: return True (all factors <= 10000 <= p, assuming p >= 10000)
    # 4. If R > 1:
    #    - If R is prime: return R <= p
    #    - If R is composite: It has a factor >= sqrt(R).
    #      If sqrt(R) > p, then it has a factor > p? Not necessarily.
    #      Example: p=100, R=101*103. Factors are 101, 103. Both > 100.
    #      Example: p=102, R=101*103. Factors 101, 103. 103 > 102.
    #      If R is composite, let d be its smallest prime factor.
    #      If d > p, then we have a factor > p.
    #      If d <= p, then we need to check the rest.
    #      So we need to find the largest prime factor of R.
    
    def has_large_factor(n, p):
        """Check if n has any prime factor > p"""
        if n <= 1:
            return False
        if p < 2:
            return True
        
        # Trial divide by small primes up to min(p, 100000)
        for sp in small_primes:
            if sp > p:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return False
        
        if n == 1:
            return False
        
        # n is now composed of primes > 100000 (or > p if p < 100000)
        # If n is prime, check if n > p
        if is_prime(n):
            return n > p
        
        # n is composite. Find if it has any factor > p
        # Use Pollard Brent to find a factor
        while n > 1:
            if is_prime(n):
                if n > p:
                    return True
                else:
                    break
            d = pollard_brent(n)
            if d == n:
                continue
            if d > p:
                return True
            n //= d
            while n % d == 0:
                n //= d
        
        return False

    best_k = 0
    best_p = 2
    
    # Search large primes
    # Use a random search strategy
    # We know that for small p, runs are short.
    # Let's try p in range [10^6, 10^9]
    # But we need to find primes.
    # Generating primes up to 10^9 is heavy.
    # We can generate random odd numbers and check primality.
    
    # Also, we should check a few small primes just in case, but we know they are bad.
    # Let's focus on large p.
    
    # To maximize k, we should check many candidates.
    # The probability of a long run decreases exponentially with k.
    # So we should check many p's for small k, and fewer for large k.
    # But we don't know the distribution.
    
    # Let's iterate through random primes.
    # We'll use a simple method: pick random 30-bit numbers, make them odd, check prime.
    
    # To ensure coverage, we might also want to check a grid of primes?
    # No, random is better for finding rare events.
    
    # Let's run for a few seconds, checking random primes.
    # We'll also check a specific range if time permits.
    
    # First, check a few known small primes to establish a baseline (though likely low)
    # Actually, the seed program found k=2 for p<2000.
    # Let's start with random search.
    
    # We'll use a deadline guard.
    while time.time() - start_time < DEADLINE:
        # Generate a random prime
        # Try a few times to find a prime
        found_prime = False
        for _ in range(100):
            # Random number in [10^6, 10^9]
            # log10(10^6) = 6, log10(10^9) = 9
            # Let's pick random digits
            # Or just random.randint(10**6, 10**9)
            # But 10^9 is large.
            # Let's mix ranges.
            if random.random() < 0.5:
                p = random.randint(10**6, 10**7)
            else:
                p = random.randint(10**7, 10**9)
            
            p |= 1  # Make odd
            
            if is_prime(p):
                found_prime = True
                break
        
        if not found_prime:
            continue
        
        # Check runs for this p
        # We want to maximize k.
        # We'll check up to k=200.
        # But we can stop early if we find a factor > p.
        
        current_k = 0
        # p^2 + 1 ... p^2 + k
        # We can check sequentially.
        # Optimization: if p^2 + j has a large factor, stop.
        
        # To speed up, we can check a few values first.
        # If they pass, we are more likely to get a long run.
        
        # Let's check up to 200, but break early.
        # We'll use a helper to check if p^2 + i is p-smooth.
        
        # To avoid recalculating p^2, we store it.
        p2 = p * p
        
        # We'll check i from 1 to 200
        # But we can jump? No, we need contiguous run.
        
        # Optimization: 
        # If p is large, p^2 is large.
        # has_large_factor uses trial division by small primes.
        # This is fast.
        
        # Let's try to find the max k for this p.
        # We'll use a while loop.
        
        # But we must respect the time limit.
        # If we spend too much time on one p, we might miss better ones.
        # So we should limit the time per p?
        # Or just check the first few k's?
        # No, we need to find the MAX k.
        
        # Strategy:
        # Check k=1. If fail, next p.
        # Check k=2. If fail, next p.
        # ...
        # This is inefficient if we restart from k=1 for each p.
        
        # Better: For each p, find the full run length.
        # But if the run is long, it takes time.
        # If the run is short, it's fast.
        
        # Let's estimate time.
        # has_large_factor involves trial division by ~1000 primes.
        # This is very fast.
        # Pollard Brent is only called if trial division fails to reduce n to 1 or a small prime.
        # For random numbers, trial division by small primes often reduces them significantly.
        
        # So checking k=1..200 for one p should be fast (< 1ms).
        # So we can check many p's.
        
        # Let's do it.
        
        # However, we must be careful: if p is small, p^2 + i might have small factors.
        # If p is large, p^2 + i is large.
        
        # Let's check the run.
        k = 0
        while k < 200:
            val = p2 + k + 1
            if has_large_factor(val, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            # Update best
            # We can print for debugging, but not required.
            pass
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
