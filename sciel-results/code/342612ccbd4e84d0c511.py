import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit

    # 1. Generate a list of small primes up to 10^6 for fast trial division
    #    We need primes up to a certain limit for the smoothness check.
    #    Since p can be up to 10^9, we cannot sieve up to p.
    #    We will use small primes for trial division, then MR/Pollard for the rest.
    
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

    # 3. Helper: Pollard-Brent factorization
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
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
        return g

    def factor(n):
        factors = []
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    # 4. Helper: Check if n is p-smooth (all prime factors <= p)
    #    Optimization: 
    #    1. Trial divide by small primes up to min(p, 10^6).
    #    2. If remainder is 1, return True.
    #    3. If remainder is prime and <= p, return True.
    #    4. If remainder is composite, factor it. If any factor > p, return False.
    def is_p_smooth(n, p):
        if n == 1:
            return True
        
        temp = n
        # Trial division by small primes
        # We only need to divide by primes <= min(p, 10^6)
        # Since small_primes is sorted, we can break when q > p or q*q > temp
        for q in small_primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        
        if temp == 1:
            return True
        
        # If temp is prime, check if it's <= p
        if is_prime_mr(temp):
            return temp <= p
        
        # If temp is composite, factor it
        # Note: temp < n <= p^2 + 200. 
        # If p is large, temp might have large factors.
        # We factor temp and check all factors.
        try:
            factors = factor(temp)
            for f in factors:
                if f > p:
                    return False
            return True
        except:
            # If factorization fails or is too slow, assume not smooth (conservative)
            # Or we can do a more careful check?
            # For now, return False to avoid false positives.
            return False

    # 5. Search Strategy:
    #    We want to maximize k.
    #    k is the length of the run p^2+1, ..., p^2+k that is p-smooth.
    #    
    #    Heuristic: Larger p allows for larger smooth numbers?
    #    Actually, for a fixed p, the probability that p^2+i is p-smooth decreases as i increases?
    #    Or does it?
    #    The density of p-smooth numbers around x is rho(log(x)/log(p)).
    #    For x ~ p^2, log(x)/log(p) ~ 2.
    #    So the probability is roughly constant?
    #    
    #    However, we know that small p have limited k.
    #    For p=2, p^2=4. 5 is not 2-smooth. k=0.
    #    For p=3, p^2=9. 10=2*5 (5>3). k=0.
    #    For p=5, p^2=25. 26=2*13 (13>5). k=0.
    #    For p=7, p^2=49. 50=2*5^2 (OK). 51=3*17 (17>7). k=1.
    #    
    #    Let's search primes in DESCENDING order?
    #    Why? Because if we find a large k with a large p, we can stop?
    #    No, a smaller p might have a larger k?
    #    Actually, it's unlikely that a small p has a very large k because the numbers are small and the constraint "factors <= p" is tight.
    #    Wait, if p is small, p^2 is small. The factors of p^2+i are small.
    #    If p=100, p^2=10000. Factors of 10001..10200.
    #    If p=10000, p^2=10^8. Factors of 10^8+1..
    #    
    #    Let's look at the "smoothness" probability.
    #    The probability that a random integer n is p-smooth is approx 1/ln(p) for n ~ p^2?
    #    Actually, Dickman function rho(u) where u = ln(n)/ln(p).
    #    If n ~ p^2, u ~ 2. rho(2) ~ 0.306.
    #    So roughly 30% of numbers are p-smooth.
    #    The expected run length is 1/(1-0.306) ~ 1.4.
    #    So k is usually small.
    #    To get k=200, we need a very lucky streak.
    #    The probability of a run of length k is (0.3)^k.
    #    For k=10, (0.3)^10 ~ 5e-6.
    #    For k=20, (0.3)^20 ~ 3e-11.
    #    So we need to test many p's to find a long run.
    #    
    #    How many p's can we test?
    #    If each test takes 1ms, we can test 100,000 p's in 100s.
    #    If we test 100,000 p's, and the probability of k>=10 is 5e-6, we expect 0.5 hits.
    #    So we might find k=10 or 11.
    #    
    #    The current program found k=7 (score 2.33 implies k=7 if seed was k=3? No, score is value/seed.
    #    If seed score is 1.0, and value is k, then seed k=3?
    #    Current score 2.3333. If seed k=3, then current k=7.
    #    Let's assume the goal is to maximize k.
    
    best_k = 0
    best_p = 3
    
    # We will search primes in a specific range.
    # Small primes are fast to check but unlikely to have long runs.
    # Large primes are slow to check (factorization) but might have long runs?
    # Actually, the smoothness probability is similar.
    # But factorization of large numbers is slow.
    
    # Let's try a hybrid:
    # 1. Check all primes up to 10^5 with fast trial division (since p <= 10^5, we can just check if remainder is 1 or prime <= p).
    # 2. Then check random primes in [10^5, 10^9] with Pollard-Brent.
    
    # Phase 1: Primes up to 10^5
    # We have small_primes up to 10^6.
    # For p <= 10^5, we can use the same is_p_smooth logic, but it will be fast because trial division works well.
    
    for p in small_primes:
        if p > 100000:
            break
        if time.time() > deadline:
            break
            
        base = p * p
        k = 0
        for offset in range(1, 201):
            if time.time() > deadline:
                break
            n = base + offset
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
    # Phase 2: Larger primes
    # We will sample primes from the range [10^5, 10^9].
    # We need to generate primes in this range.
    # We can use a segmented sieve or just random search.
    # Random search: pick random x, check if prime, then check smoothness.
    # This is inefficient because prime density is low.
    # Better: Use a segmented sieve to generate primes in chunks.
    
    # Let's implement a simple segmented sieve for the range [10^5, 10^6] first, then expand?
    # Or just use random primes?
    # Given the time constraint, let's try to search a few large primes.
    
    # Actually, let's refine the search for Phase 1 to be faster.
    # The is_p_smooth function for small p is fast.
    # For p > 10^5, is_p_smooth is slow due to Pollard-Brent.
    
    # Let's estimate how many large primes we can test.
    # If each takes 10ms, we can test 1000 primes in 10s.
    # If we test 1000 primes, and P(k>=10) is 5e-6, we expect 0.005 hits.
    # So we are unlikely to beat k=7 with large primes unless we get lucky or the probability is higher.
    
    # Is the probability higher for large p?
    # rho(2) is constant.
    # But maybe the "correlation" between consecutive numbers is different?
    # Unlikely.
    
    # So the best strategy might be to search MORE small primes?
    # We searched up to 10^5.
    # Can we search up to 10^6?
    # For p=10^6, trial division is 78k primes.
    # For each n, we do 78k divisions.
    # 200 n's * 78k divisions = 15 million divisions per p.
    # 15 million divisions take ~1s in Python.
    # So we can test ~100 primes in 100s.
    # But we already tested up to 10^5 (9500 primes).
    # If we test 100 more primes in [10^5, 10^6], the chance of finding a better k is low.
    
    # Wait, did we search ALL primes up to 10^5?
    # Yes, the loop `for p in small_primes: if p > 100000: break` does that.
    # So best_k is the max k for p <= 10^5.
    
    # If best_k is 7, can we do better?
    # Maybe p > 10^5 has a better k?
    # Let's try to test a few primes in [10^5, 10^6] with the fast trial division method.
    # We can optimize is_p_smooth for p <= 10^6 by precomputing primes up to p?
    # No, we already have small_primes up to 10^6.
    # So for p <= 10^6, is_p_smooth is fast (trial division only).
    
    # Let's extend Phase 1 to p <= 10^6.
    # But we need to be careful with time.
    # We already spent time on p <= 10^5.
    # Let's check time.
    
    if time.time() < deadline - 5.0:
        # Continue searching primes in [10^5, 10^6]
        for p in small_primes:
            if p > 1000000:
                break
            if p <= 100000:
                continue # Already checked
            
            if time.time() > deadline:
                break
            
            base = p * p
            k = 0
            for offset in range(1, 201):
                if time.time() > deadline:
                    break
                n = base + offset
                if is_p_smooth(n, p):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # If we have time, try some large primes with Pollard-Brent
    # But this is risky.
    
    return {'p': best_p, 'k': best_k}
