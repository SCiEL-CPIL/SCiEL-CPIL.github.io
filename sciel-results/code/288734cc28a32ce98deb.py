import math
import time
import random
import sys

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
    # 1. We want to find a prime p such that p^2 + 1 ... p^2 + k are all p-smooth.
    # 2. This means for each i in 1..k, p^2 + i has no prime factor q > p.
    # 3. If q > p is a prime factor of p^2 + i, then p^2 ≡ -i (mod q).
    # 4. We can sieve out primes p that are "bad" for any i in 1..200.
    # 5. However, we only need to check q up to some limit. If q > p^2 + 200, it can't be a factor.
    #    But q > p is the condition for "not p-smooth".
    #    So we need to check if there exists a prime q > p such that q | (p^2 + i).
    #    Since p^2 + i < p^2 + 200, any factor q must be <= p^2 + 200.
    #    If we sieve for q up to Q_max, we might miss factors q > Q_max.
    #    But if we choose Q_max large enough, we cover most cases.
    #    Actually, if p is small, p^2 is small. If p is large, p^2 is large.
    
    # Let's try a hybrid approach:
    # 1. Sieve primes p up to P_LIMIT.
    # 2. For each candidate p, we can quickly check if it's "likely" to be smooth
    #    by checking if p^2 + i has any small prime factor > p? No, that's the opposite.
    #    We need to check if p^2 + i has ANY prime factor > p.
    
    # The inspiration suggests sieving p based on q.
    # Let's iterate q (primes) and mark p as bad if p^2 ≡ -i mod q for any i in 1..200.
    # We need to do this for all q > p.
    # This is tricky because the condition "q > p" depends on p.
    
    # Alternative:
    # For a fixed p, we need to ensure that for all i in 1..200, p^2 + i is p-smooth.
    # This is equivalent to: for all i in 1..200, all prime factors of p^2 + i are <= p.
    
    # Let's stick to the direct search but optimize the smoothness check.
    # The current program does a sequential scan.
    # Let's try scanning primes in a different order or using a pre-filter.
    
    # Pre-filter idea:
    # If p^2 + 1 is prime and > p, then k=0.
    # If p^2 + 1 is not prime, it has a factor < p^2+1.
    # If that factor is > p, then k=0.
    # So for k>=1, p^2+1 must be composite with all factors <= p, OR p^2+1 <= p (impossible for p>=2).
    # So p^2+1 must be p-smooth.
    
    # Let's generate primes up to 10^6.
    sieve_limit = 10**6
    primes = sieve(sieve_limit)
    
    # We can also try some larger primes if time permits, but the probability drops.
    # Let's focus on finding the best k among primes <= 10^6.
    
    # To speed up, we can use a "fast fail" check.
    # For a given p, check i=1, 2, ... until we find a non-smooth number.
    # We can check i=1 first. If p^2+1 is not p-smooth, skip p.
    # This is a very strong filter.
    
    # Let's implement a faster is_p_smooth(n, p) function.
    # It should return True if all prime factors of n are <= p.
    
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        # If n is prime and n > p, return False
        if is_prime(n):
            return n <= p
        
        # Trial division by primes up to min(p, sqrt(n))
        # If we find a factor > p, return False
        # If we reduce n to 1, return True
        # If remaining n > 1, it must be prime (or product of primes > sqrt(n)? No, if we divide by all primes <= sqrt(n), remaining is prime).
        # Wait, if we divide by primes up to p, and p < sqrt(n), we might miss a factor.
        # Example: n = 100, p = 5. sqrt(100)=10. Primes <= 5: 2, 3, 5.
        # 100 / 2 = 50, 50 / 2 = 25, 25 / 5 = 5, 5 / 5 = 1. Smooth.
        # Example: n = 14, p = 3. sqrt(14)~3.7. Primes <= 3: 2, 3.
        # 14 / 2 = 7. 7 is prime. 7 > 3. Not smooth.
        
        # So:
        # 1. Divide n by all primes <= min(p, sqrt(n)).
        # 2. If n > 1, check if n is prime and n <= p.
        #    If n is prime and n > p, not smooth.
        #    If n is composite, it must have a factor <= sqrt(n).
        #    But we already divided by all primes <= min(p, sqrt(n)).
        #    If p >= sqrt(n), then we divided by all primes <= sqrt(n), so n must be 1 or prime.
        #    If p < sqrt(n), we stopped dividing at p.
        #    Then n might have factors > p.
        #    In that case, we need to check if n has any prime factor > p.
        #    This is equivalent to checking if n is p-smooth.
        
        # Efficient check:
        # Let's just factorize n and check max factor.
        # Factorization of n ~ p^2.
        # If p is small, n is small.
        # If p is large, n is large.
        
        # Let's use the factorize function but with a cutoff?
        # No, factorize is fast enough for 10^12.
        
        # Optimization: Check small primes first.
        # If n is divisible by a prime q > p, return False.
        # We can iterate primes q from small to large.
        # If q > p, stop. If n % q == 0, return False.
        # If q > sqrt(n), stop.
        # If loop finishes and n > 1, then n is prime. Check n <= p.
        
        # This avoids full factorization if a large factor is found early.
        
        # Get primes up to min(p, sqrt(n))
        # We have a global list of primes? No, we generated primes up to 10^6.
        # If p < 10^6, we have primes up to p.
        # If p >= 10^6, we might not have primes up to sqrt(n) if n is large.
        # But for p <= 10^6, n <= 10^12 + 200. sqrt(n) <= 10^6.
        # So for p <= 10^6, we have all primes up to sqrt(n).
        
        # So for p <= 10^6:
        # Iterate primes q in primes_list.
        # If q > p: break (if n > 1, n is prime factor > p? No, n might be composite with factors > p? No, if we divided by all primes <= p, and p >= sqrt(n), then n is 1 or prime. If p < sqrt(n), we stop at p. Then n might have factors > p. But we only need to know if ANY factor > p. If n > 1 after dividing by primes <= p, and p < sqrt(n), it's possible n is prime > p or composite with factors > p. In both cases, it's NOT p-smooth. So if n > 1 and we stopped because q > p, return False. If we stopped because q > sqrt(n), then n is prime. Check n <= p.)
        
        # Algorithm for is_p_smooth(n, p) for p <= 10^6:
        # 1. For each prime q in primes_list:
        #    if q > p: return False (if n > 1) -- wait, if n==1 return True.
        #    if q * q > n: break
        #    while n % q == 0: n //= q
        # 2. After loop:
        #    if n == 1: return True
        #    # n is prime
        #    return n <= p
        
        # This is very fast!
        
        for q in primes:
            if q > p:
                return n == 1
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        if n == 1:
            return True
        return n <= p

    # Now, iterate through primes p.
    # We can optimize the order.
    # Maybe larger p are more likely to have longer runs?
    # Or smaller p?
    # Let's just iterate.
    
    # To save time, we can check p in increasing order.
    # If we find a good k, we can try to beat it.
    
    # Let's also try a random sample of larger primes if we have time.
    
    # First pass: all primes <= 10^6
    for p in primes:
        if time.time() > deadline:
            break
            
        p2 = p * p
        
        k = 0
        # Check i = 1 to 200
        # We can break early if k > best_k and we just want to know if it's better?
        # No, we need the max k.
        # But if we find k > best_k, we update.
        # If we find a non-smooth number at i <= best_k, we can skip the rest for this p?
        # Yes, if p^2 + i is not smooth for i <= best_k, then k(p) < i <= best_k, so it can't beat best_k.
        
        # So:
        # Check i = 1 to best_k.
        # If any is not smooth, continue to next p.
        # If all smooth, check i = best_k + 1, ... until non-smooth or 200.
        
        valid = True
        for i in range(1, best_k + 1):
            if not is_p_smooth(p2 + i, p):
                valid = False
                break
        
        if not valid:
            continue
            
        current_k = best_k
        while current_k < 200:
            current_k += 1
            if not is_p_smooth(p2 + current_k, p):
                break
            if current_k > best_k:
                best_k = current_k
                best_p = p
                
        if time.time() > deadline:
            break

    # If we have time, try some larger primes?
    # The probability of long runs drops, but maybe we can find a lucky one.
    # Let's try primes between 10^6 and 10^7?
    # Sieve is expensive.
    # We can use Miller-Rabin to generate primes.
    
    if time.time() < deadline - 10.0:
        # Try a few random large primes
        # Or scan a range
        # Let's scan primes from 10^6 to 10^6 + 10^5
        # We need to generate primes in this range.
        # Simple way: check each odd number with is_prime.
        
        start_p = 10**6
        end_p = 10**6 + 10**5
        # Ensure start_p is odd
        if start_p % 2 == 0:
            start_p += 1
            
        for p in range(start_p, end_p, 2):
            if time.time() > deadline:
                break
            if not is_prime(p):
                continue
            
            p2 = p * p
            valid = True
            for i in range(1, best_k + 1):
                # Note: is_p_smooth uses 'primes' list up to 10^6.
                # If p > 10^6, the list is not sufficient for the "q > p" check logic if p < sqrt(n).
                # But for p ~ 10^6, n ~ 10^12, sqrt(n) ~ 10^6.
                # So primes list up to 10^6 is sufficient for sqrt(n).
                # The logic "if q > p: return n == 1" works because if p > 10^6, we run out of primes in the list before q > p?
                # No, the list ends at ~10^6.
                # If p = 10^6 + 100, and we need to check factors up to p, we only have primes up to 10^6.
                # We might miss a prime factor q where 10^6 < q <= p.
                # If such q divides n, and we didn't check it, we might incorrectly say n is smooth?
                # No, if we don't divide by q, n remains divisible by q.
                # At the end, n > 1. We check n <= p.
                # If n is composite and has a factor q > 10^6, we don't know.
                # So is_p_smooth is only correct if we have all primes up to min(p, sqrt(n)).
                # For p > 10^6, we don't have primes up to p.
                # So we should only use this fast check for p <= 10^6.
                # For p > 10^6, we should use full factorization.
                
                # Since we are in the "large primes" section, let's use full factorization for safety.
                # Or just skip this section if it's too slow/complex.
                # Given the low probability of improvement, let's just stick to p <= 10^6.
                # The loop above covers p <= 10^6.
                
                # Let's break out of this large prime search to avoid bugs.
                break
                
            if not valid:
                continue

    if best_k == 0:
        # Fallback
        for p in [7, 11, 13, 17, 19, 23, 29, 31]:
            if is_prime(p):
                p2 = p * p
                k = 0
                while k < 200:
                    n = p2 + k + 1
                    if not is_p_smooth(n, p):
                        break
                    k += 1
                if k > 0:
                    best_k = k
                    best_p = p
                    break
        if best_k == 0:
            best_p = 7
            best_k = 1

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
