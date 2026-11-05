import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate primes up to 10^6
    limit_small = 10**6
    sieve = np.ones(limit_small + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    small_primes = np.array([i for i, is_p in enumerate(sieve) if is_p])
    
    # 2. Helper: Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        # Check small primes up to 37
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # 3. Helper: Check if n is p-smooth (all prime factors <= p)
    # Optimized using numpy for trial division by small primes.
    def is_p_smooth(n, p):
        if n <= p:
            return True
        
        # Use primes up to min(p, 10^6)
        # We need primes <= p.
        # Since p is likely < 10^6 in our search strategy, we can use small_primes directly.
        # But to be safe, let's find the index.
        
        # For speed, we'll do trial division with numpy where possible, 
        # but n is a scalar.
        
        # Strategy:
        # 1. Divide n by all small primes <= p.
        # 2. If remainder is 1, True.
        # 3. If remainder is prime and <= p, True.
        # 4. If remainder > p, False.
        
        # Optimization: 
        # If p is large, we have many primes.
        # We can vectorize the modulo operation?
        # No, n is a single integer.
        
        # Let's stick to a fast loop but break early.
        # To speed up, we can pre-filter:
        # If n is even and p >= 2, divide by 2.
        
        temp = n
        
        # Iterate primes
        # We can use the small_primes array.
        # We need to stop when q > p or q*q > temp.
        
        # Since p is a prime from small_primes, we can use its index.
        # But we don't have the index passed.
        # Let's just iterate small_primes and break if q > p.
        
        for q in small_primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
            if temp == 1:
                return True
        
        if temp == 1:
            return True
        
        # If temp > 1, it's either prime or composite.
        # If it's composite, it has a factor > p (since we divided all <= p).
        # So it must be prime and <= p to be p-smooth.
        if temp <= p:
            return True
        
        return False

    best_k = 0
    best_p = 3

    # Strategy:
    # We want to maximize k.
    # We can iterate k from 200 down to 1.
    # For each k, we search for a prime p such that p^2+1...p^2+k are p-smooth.
    # If found, return immediately? No, we might find a larger k later?
    # No, if we iterate k descending, the first valid k is the maximum.
    # But finding a p for a given k might be hard.
    
    # Alternative:
    # Iterate p, and for each p, find max k.
    # This is what the current program does.
    
    # Let's try a hybrid:
    # 1. Try to find a p for k=200.
    # 2. If not found, k=199, etc.
    # This requires checking many p's for each k.
    
    # Actually, the current program's weakness is that it checks p in increasing order.
    # If a large p has a long run, we miss it if we run out of time.
    # Also, checking k for a given p is slow.
    
    # Let's optimize the inner loop (checking p-smoothness).
    # And search p in a smarter way.
    
    # What if we precompute p-smoothness for a range of p?
    # No, n depends on p.
    
    # Let's stick to iterating p, but prioritize p that are likely to have long runs.
    # Are there heuristics?
    # If p is large, p^2 is large.
    # The probability that a random number is p-smooth decreases as p decreases relative to n.
    # Here n ~ p^2.
    # The "smoothness" threshold is p.
    # The density of p-smooth numbers around p^2 is related to the Dickman function rho(2) ~ 0.3.
    # So ~30% of numbers are p-smooth.
    # The probability of k consecutive p-smooth numbers is ~ 0.3^k.
    # For k=10, 0.3^10 ~ 6e-6.
    # We need to check ~ 1e6 numbers to find one run of 10.
    # For k=20, 0.3^20 ~ 3e-11.
    # We need ~ 3e10 numbers.
    # This suggests k is likely small, unless there's a structure.
    
    # The current score is 2.3333.
    # The seed score was 1.0.
    # The best known k might be around 14 (if seed was 6)?
    # Or maybe the seed found k=10 and current found k=14?
    # Score = value / seed.
    # If seed k=6, score 2.33 -> k=14.
    
    # Let's try to find k > 14.
    
    # To do this efficiently:
    # 1. Use a faster is_p_smooth check.
    # 2. Search p more broadly.
    
    # Fast is_p_smooth:
    # We can use a precomputed table of smallest prime factors (SPF) up to 10^6?
    # No, n is up to 10^12.
    
    # Let's try to optimize the search for p.
    # We can check multiple p's in parallel? No, single thread.
    
    # What if we check p in random order?
    # Or check p that are "smooth" themselves?
    
    # Let's try a different angle:
    # Instead of checking p^2+1, p^2+2...
    # We can check if the product (p^2+1)...(p^2+k) is p-smooth.
    # This is equivalent.
    
    # Let's refine the current approach:
    # 1. Generate primes up to 5*10^6 to allow larger p.
    # 2. For each p, check k.
    # 3. Use a faster smoothness check.
    
    # Faster smoothness check:
    # We can precompute primes up to 10^6.
    # For a given n, we divide by primes.
    # We can use a list of primes and break early.
    
    # Let's increase the sieve limit to 2*10^6.
    limit_small = 2 * 10**6
    sieve = np.ones(limit_small + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    small_primes = np.array([i for i, is_p in enumerate(sieve) if is_p])
    
    # Re-define is_p_smooth with the larger list
    def is_p_smooth_fast(n, p):
        if n <= p:
            return True
        
        temp = n
        
        # We need primes <= p.
        # Since p is from small_primes, we can find its index.
        # But we don't have the index.
        # Let's just iterate and break.
        
        # Optimization:
        # If temp becomes 1, return True.
        # If q > p, break.
        # If q*q > temp, break.
        
        for q in small_primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
            if temp == 1:
                return True
        
        if temp == 1:
            return True
        
        if temp <= p:
            return True
        
        return False

    best_k = 0
    best_p = 3

    # Search primes
    # We can't iterate all primes up to 2*10^6 if checking is slow.
    # Let's estimate time per check.
    # For p=10^6, n=10^12.
    # Trial division by primes up to 10^6.
    # ~ 78,000 primes.
    # Each check ~ 78,000 ops.
    # 110s / 78,000 ops ~ 1400 checks per second?
    # No, Python is slow.
    # Maybe 1000 checks per second.
    # 110s -> 110,000 checks.
    # If we check k=10 for each p, that's 10 checks per p.
    # So 11,000 p's.
    # This is not enough to search 2*10^6 primes.
    
    # We need a faster check.
    
    # Idea:
    # For a fixed p, we check n = p^2 + 1, p^2 + 2, ...
    # We can reuse the factorization?
    # No, they are different numbers.
    
    # What if we check p in a specific range?
    # Maybe small p has longer runs?
    # For p=2, n=5,6,7...
    # 5 (prime > 2) -> fail. k=0.
    # For p=3, n=10,11...
    # 10 (2*5, 5>3) -> fail. k=0.
    # For p=5, n=26,27...
    # 26 (2*13, 13>5) -> fail.
    
    # It seems small p fails quickly.
    # Larger p might have longer runs.
    
    # Let's try to search p in a random order or large p first?
    # If large p has a longer run, we find it faster.
    
    # Let's try p from 10^5 to 2*10^6.
    # Step size?
    
    # Let's use a different strategy:
    # Check k=200, 199...
    # For each k, sample random p's.
    # If we find a p, we can verify and return.
    
    # But random sampling might miss it.
    
    # Let's go back to iterating p, but optimize the inner loop.
    # We can use a C-extension? No.
    # We can use numpy to check divisibility by multiple primes at once?
    # No, n is a scalar.
    
    # How about this:
    # We precompute the smallest prime factor for all numbers up to 10^6?
    # This doesn't help for n ~ 10^12.
    
    # Let's try to limit the search to p < 10^6 and optimize the check.
    # If we can do 10,000 checks per second, we can check 110,000 numbers.
    # If we check k=10 for each p, we check 10 numbers per p.
    # So 11,000 p's.
    # There are 78,000 primes < 10^6.
    # We can't check all.
    
    # We need to be smarter.
    
    # What if we check p^2 + 1 first?
    # If p^2 + 1 is not p-smooth, skip p.
    # This is 1 check per p.
    # 110,000 p's.
    # This covers all primes < 10^6!
    
    # Strategy:
    # 1. For each prime p < 10^6:
    #    a. Check if p^2 + 1 is p-smooth.
    #    b. If not, skip.
    #    c. If yes, check p^2 + 2, p^2 + 3, ... until fail.
    
    # This is much more efficient!
    # Most p will fail at step (a).
    
    # Let's implement this.
    
    for p in small_primes:
        if time.time() > deadline:
            break
        
        # Check p^2 + 1
        n1 = p * p + 1
        if not is_p_smooth_fast(n1, p):
            continue
        
        # If p^2 + 1 is p-smooth, check further
        k = 1
        for offset in range(2, 201):
            if time.time() > deadline:
                break
            
            n = p * p + offset
            if not is_p_smooth_fast(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
