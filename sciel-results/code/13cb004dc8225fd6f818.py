import math
import random

# Pre-compute primes up to 10^6 using Sieve of Eratosthenes
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

PRIMES = sieve(10**6)

def phi(n):
    if n == 1:
        return 1
    result = n
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
    if n > 1:
        result -= result // n
    return result

def construct():
    # We want to maximize k.
    # The maximum possible k is 200.
    # We iterate k from 200 down to 1.
    # For each k, we try to find a start s.
    
    # To improve chances, we can use a deterministic set of "good" starting points
    # or random ones. Given the 120s limit, we should be efficient.
    # Trial division up to 10^6 is fast.
    # phi(n) for n ~ 10^12 takes ~ 10^6 / log(10^6) ~ 70000 operations? No, 
    # number of primes up to 10^6 is ~ 78498.
    # Each phi call is ~ 80k iterations worst case.
    # 200 phi calls = 16M iterations.
    # Python is slow. 16M simple ops might take 1-2 seconds.
    # We can do ~50-100 checks per second?
    # 120 seconds * 50 checks = 6000 checks.
    # Each check involves k phi computations.
    # If k=200, each check is 200 phi calls.
    # So ~30 full checks of k=200 in 120s.
    # We need to be smart about which s to try.
    
    # Strategy:
    # 1. Try s = 10^12 - k + 1 - offset for various offsets.
    # 2. Try random s in [10^11, 10^12].
    
    # However, if k=200 is rare, we might spend all time on k=200 and fail.
    # We should balance.
    # But the score is value/seed. If we get k=100, score is 100/seed.
    # If we get k=200, score is 200/seed.
    # It's better to try for high k.
    
    # Let's add a time limit per k or overall?
    # The function must return within 120s.
    
    # We will try a mix of candidates.
    # To avoid repeating work, we can cache phi values? 
    # The ranges are disjoint mostly.
    
    for k in range(200, 0, -1):
        # Generate candidates
        # We want s + k - 1 <= 10^12 => s <= 10^12 - k + 1
        max_s = 10**12 - k + 1
        
        # Candidate 1: Near the top
        s1 = max_s
        # Candidate 2: Near the top - 100
        s2 = max_s - 100
        # Candidate 3: Near the top - 1000
        s3 = max_s - 1000
        # Candidate 4: Random in upper range
        s4 = random.randint(10**11, max_s)
        # Candidate 5: Random in mid range
        s5 = random.randint(10**10, max_s)
        
        candidates = [s1, s2, s3, s4, s5]
        
        # Deduplicate and filter
        seen_starts = set()
        valid_candidates = []
        for s in candidates:
            if s >= 1 and s not in seen_starts:
                seen_starts.add(s)
                valid_candidates.append(s)
                
        # Check candidates
        for s in valid_candidates:
            seen_phis = set()
            valid = True
            for i in range(k):
                p_val = phi(s + i)
                if p_val in seen_phis:
                    valid = False
                    break
                seen_phis.add(p_val)
            
            if valid:
                return {'start': s, 'length': k}
                
        # If we fail for this k, we might want to try more random candidates?
        # Given time constraints, we just drop k.
        # But maybe for k=200, we should try more?
        # Let's add a loop for a few more random candidates if k is large.
        if k > 150:
            for _ in range(10):
                s = random.randint(10**10, max_s)
                if s in seen_starts:
                    continue
                seen_starts.add(s)
                seen_phis = set()
                valid = True
                for i in range(k):
                    p_val = phi(s + i)
                    if p_val in seen_phis:
                        valid = False
                        break
                    seen_phis.add(p_val)
                if valid:
                    return {'start': s, 'length': k}

    return {'start': 1, 'length': 1}
