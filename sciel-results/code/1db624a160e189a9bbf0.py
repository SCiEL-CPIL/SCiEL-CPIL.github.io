import math
import time
import random

# Deterministic Miller-Rabin bases for n < 2^64
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    # Small primes check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
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
    
    # Random parameters
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
        g = 1
        r = 1
        q = 1
        x = y
        ys = y
        
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
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
    factors = []
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            factors.append(x)
        else:
            d = pollard_brent(x)
            stack.append(d)
            stack.append(x // d)
    return factors

def largest_prime_factor(n):
    if n <= 1:
        return 1
    factors = factorize(n)
    return max(factors)

def sieve(limit):
    if limit < 2:
        return []
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_arr[j] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_p = 0
    best_k = 0
    
    # Strategy:
    # 1. Try a range of primes. Since checking k up to 200 involves factoring numbers around p^2,
    #    Pollard-Brent is fast for these sizes.
    # 2. We can try primes from small to large, or randomize.
    #    The "long runs" property is rare. Larger p might have longer runs?
    #    Actually, the density of p-smooth numbers near p^2 is roughly related to the Dickman function.
    #    For x = p^2 + i, the probability that all factors are <= p is roughly rho(2) ~ 0.306? No, that's for x^(1/2).
    #    If we look at p^2, the "smoothness bound" is p = sqrt(p^2).
    #    The probability that a random number near N is N^{1/2}-smooth is rho(2) ≈ 0.306.
    #    The probability that k consecutive numbers are all p-smooth is roughly (0.306)^k.
    #    For k=10, this is ~ 0.306^10 ≈ 6e-6.
    #    For k=20, ~ 1e-11.
    #    So finding k=20 requires checking ~10^11 candidates. That's too many.
    #    However, the numbers are not random; they are consecutive integers near p^2.
    #    There is a known result about "smooth runs".
    #    
    #    Let's just search. We can generate primes up to a limit and test them.
    #    Limit for p: if p is too large, p^2 is huge, but Pollard-Brent handles 64-bit ints well.
    #    Let's try primes up to 10^6 first. If time remains, go higher.
    
    # Generate primes up to 10^5 first for a quick scan
    # Then expand if needed.
    
    # Actually, let's just iterate through primes.
    # We need an efficient way to get primes.
    # Sieve up to 10^6 gives 78498 primes.
    # Testing each prime involves up to 200 factorizations.
    # 78498 * 200 = 15.7 million factorizations.
    # Each factorization of a ~10^12 number takes ~1ms? Maybe less.
    # 15.7s. Feasible.
    
    # Let's sieve up to 10^6.
    sieve_limit = 10**6
    primes = sieve(sieve_limit)
    
    # Optional: If we have time, we could try larger primes, but the probability of long runs drops.
    # However, maybe specific primes have better properties? Unlikely to be systematic without deeper number theory.
    
    # We will test primes in increasing order.
    # To maximize k, we should probably check all primes in the range.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Check run length
        k = 0
        # p^2 + 1 to p^2 + k
        # We want the largest k such that all are p-smooth.
        # Note: p^2 + 1 is always coprime to p.
        
        # Optimization: 
        # If p^2 + j has a factor > p, stop.
        
        # We can cache factorizations? No, numbers are distinct.
        
        # Let's check up to k=200.
        # Early exit if k > best_k? No, we want to find the max.
        # But if we find k > best_k, we update.
        
        # To save time, we can stop checking for this p once k reaches a value that cannot beat best_k?
        # No, we need the full k for this p to know if it's the new best.
        # But if we've already reached 200, we can stop (max k is 200).
        
        # Actually, if best_k is already high, say 15, and we find k=15 for current p,
        # we still need to check if it goes to 16, 17...
        
        # Let's just run the loop.
        
        # Pre-calculate p^2
        p2 = p * p
        
        # Check consecutive integers
        # We can break early if k is already > best_k and we want to verify?
        # No, just find the actual k.
        
        # Optimization: if k > best_k, we can potentially stop early if we hit a non-smooth number?
        # No, we need to find the exact k to report.
        
        # However, if we find a run of length L, and L <= best_k, we don't need to check further?
        # Yes! If we find that p^2 + j is NOT p-smooth for some j <= best_k, then k < best_k, so it can't be the new best.
        # We only need to check beyond best_k if all previous ones were smooth.
        
        # So:
        # Check j = 1 to best_k.
        # If any is not smooth, skip this p.
        # If all are smooth, continue checking j = best_k+1, best_k+2... until we find a non-smooth one or reach 200.
        
        # This saves time if best_k grows.
        
        # But initially best_k = 0.
        
        # Let's implement this logic.
        
        # First pass: check 1 to best_k
        valid_so_far = True
        for j in range(1, best_k + 1):
            n = p2 + j
            # Check if n is p-smooth
            # We need to factor n and check max factor <= p
            # If n is prime and n > p, fail.
            # If n has a prime factor > p, fail.
            
            # Fast check: if n > p and is_prime(n): fail
            if n > p and is_prime(n):
                valid_so_far = False
                break
            
            # Otherwise, factorize
            # Optimization: if n is small, trial division?
            # Pollard-Brent is good.
            
            # We can also check small factors first?
            # The factorization function handles it.
            
            factors = factorize(n)
            if max(factors) > p:
                valid_so_far = False
                break
        
        if not valid_so_far:
            continue
        
        # All 1..best_k are smooth.
        # Now extend k
        current_k = best_k
        while current_k < 200:
            current_k += 1
            n = p2 + current_k
            
            if n > p and is_prime(n):
                break
            
            factors = factorize(n)
            if max(factors) > p:
                break
            
            if current_k > best_k:
                best_k = current_k
                best_p = p
                
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    # If we haven't found anything, return a valid small case
    # The seed program found k=2 for some p.
    # Let's ensure we return a valid result.
    # If best_k == 0, we need to find at least k=1.
    # p^2+1 is always > 1.
    # For p=3, p^2+1 = 10 = 2*5. 5 > 3. Fail.
    # For p=5, p^2+1 = 26 = 2*13. 13 > 5. Fail.
    # For p=7, p^2+1 = 50 = 2*5^2. Max factor 5 <= 7. OK. k=1.
    # p^2+2 = 51 = 3*17. 17 > 7. Fail. So k=1 for p=7.
    
    if best_k == 0:
        # Fallback: find any valid p
        # Try small primes again with simple logic if sieve failed or something
        for p in [7, 11, 13, 17, 19, 23, 29, 31]:
            if is_prime(p):
                p2 = p * p
                k = 0
                while k < 200:
                    n = p2 + k + 1
                    if n > p and is_prime(n):
                        break
                    factors = factorize(n)
                    if max(factors) > p:
                        break
                    k += 1
                if k > 0:
                    best_k = k
                    best_p = p
                    break
        
        if best_k == 0:
            # Should not happen, but safe fallback
            best_p = 7
            best_k = 1

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
