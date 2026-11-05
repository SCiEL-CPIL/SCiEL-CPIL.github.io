import math
import random
import sys

# EVOLVE-BLOCK-START

# Pre-compute small primes for trial division and filtering
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

SMALL_PRIMES = sieve(10000) # Primes up to 10^4

def is_prime_miller_rabin(n):
    if n < 2: return False
    if n in (2, 3): return True
    if n % 2 == 0: return False
    # Deterministic for n < 2^64 with specific bases
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n: continue
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

def pollard_rho(n):
    if n % 2 == 0: return 2
    if n % 3 == 0: return 3
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

def factor(n):
    factors = []
    for p in SMALL_PRIMES:
        if p * p > n: break
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        if is_prime_miller_rabin(n):
            factors.append(n)
        else:
            # Pollard Rho
            f = pollard_rho(n)
            factors.extend(factor(f))
            factors.extend(factor(n // f))
    return factors

def phi(n):
    if n == 1: return 1
    orig_n = n
    result = n
    for p in set(factor(n)):
        result -= result // p
    return result

def construct():
    MAX_N = 10**12
    MAX_K = 200
    
    best_k = 1
    best_s = 1
    
    # Try to find a long run
    # We start with a high target k and decrease if not found quickly
    # Or we can just try random s and see what max k we get
    
    # To maximize k, we should search for s that works for large k.
    # Let's iterate k from high to low, and for each k, try some random s.
    
    # Time limit consideration: 120s.
    # Let's try k=100, 90, 80...
    
    import time
    start_time = time.time()
    time_limit = 100 # Leave buffer
    
    for target_k in range(100, 0, -5):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        trials = 0
        while not found and trials < 500 and time.time() - start_time < time_limit:
            trials += 1
            # Pick random start
            # Bias towards larger numbers?
            # Large numbers have larger phi, less collision?
            s = random.randint(1, MAX_N - target_k)
            
            # Quick check: do all s+i have a large prime factor?
            # If yes, very likely distinct.
            # If no, still check, but less likely.
            
            phis = []
            all_large = True
            for i in range(target_k):
                n = s + i
                # Compute phi
                p_val = phi(n)
                phis.append(p_val)
                
                # Check for large prime factor
                # We can reuse factorization? 
                # phi() calls factor(). Let's optimize by getting factors once.
            
            # Optimization: get factors for all n first?
            # No, just check distinctness.
            
            if len(set(phis)) == target_k:
                if target_k > best_k:
                    best_k = target_k
                    best_s = s
                found = True
                # If we found a run of length target_k, we can try to extend it?
                # Or just accept and try higher k? 
                # Since we are iterating down, we found a valid k.
                # We can break to try smaller k? No, we want MAX k.
                # So if we found k=100, we should stop and return 100?
                # Or try to see if k=105 works?
                # Let's just return the first found high k.
                # But maybe a slightly lower k is easier to find?
                # No, if k=100 works, k=99 also works (subset).
                # So finding k=100 is better than k=99.
                # We should try to find the MAXIMUM possible k.
                # So we should search for k=200, then 199, etc.
                # But 200 might be impossible.
                # So we should do a binary search or linear search from high to low.
                # If we find k=100, we know max is at least 100.
                # We can then try 150. If fail, try 125.
                
                # Let's refine: 
                # 1. Try k=100. If found, best_k=100.
                # 2. Try k=150. If found, best_k=150.
                # 3. Try k=125. If found, best_k=125.
                # This is inefficient.
                
                # Better: 
                # Start with k=100. 
                # If found, set lower_bound = 100. 
                # Then try k=150. 
                # If found, lower_bound = 150. 
                # Then try k=175, 187, 193, 196, 198, 199, 200.
                
                # Let's restructure the loop.
                break
            
        # If not found for this target_k, continue to next lower target_k
        
    # If we didn't find anything high, we might have missed.
    # Let's do a second pass if best_k is low.
    
    if best_k < 50:
        for target_k in range(20, 0, -1):
            if time.time() - start_time > time_limit:
                break
            found = False
            trials = 0
            while not found and trials < 1000 and time.time() - start_time < time_limit:
                trials += 1
                s = random.randint(1, MAX_N - target_k)
                phis = [phi(s+i) for i in range(target_k)]
                if len(set(phis)) == target_k:
                    if target_k > best_k:
                        best_k = target_k
                        best_s = s
                    found = True
                    break
    
    return {'start': int(best_s), 'length': int(best_k)}

# EVOLVE-BLOCK-END
