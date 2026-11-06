"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

# Increase recursion limit for safety, though we use iterative factorization
sys.setrecursionlimit(10000)

# --- Primality Testing ---

# Deterministic Miller-Rabin for n < 2^64
# Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
# This is sufficient for n < 3,317,044,064,679,887,385,961,981
# Our n is at most (10^9)^2 + 200 = 10^18 + 200, which is well within range.

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    # Miller-Rabin
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    for a in _SMALL_PRIMES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# --- Factorization ---

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    # Use a random polynomial x^2 + c
    # We'll try a few different c's if it fails
    for c in [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        x = random.randint(2, n - 1)
        y = x
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d
    return n # Should rarely happen

def largest_prime_factor(n):
    if n == 1:
        return 1
    
    max_f = 1
    stack = [n]
    
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        
        if is_prime(m):
            if m > max_f:
                max_f = m
            continue
        
        # Try small primes first to strip them out
        for p in _SMALL_PRIMES:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                if m == 1:
                    if p > max_f:
                        max_f = p
                    continue
                # m is no longer divisible by p
                # Re-evaluate m
                break
        
        # If we broke out of the small prime loop with m > 1, m is not divisible by small primes
        # Check if m is prime
        if m > 1:
            if is_prime(m):
                if m > max_f:
                    max_f = m
            else:
                # Use Pollard Rho
                d = pollard_rho(m)
                if d == m:
                    # Pollard Rho failed? Try again or fallback?
                    # For robustness, if it returns n, we treat it as prime (error) or retry.
                    # Given deterministic bases, if is_prime says composite, it should be.
                    # Let's retry pollard_rho with different random seeds implicitly by calling again
                    # But pollard_rho above has fixed c's. Let's just push m back? No, infinite loop.
                    # Let's assume Pollard Rho works. If not, we might hang.
                    # To be safe, if d==m, we'll just assume it's a large prime factor for now?
                    # No, that's dangerous. Let's add a retry logic in pollard_rho if it returns n.
                    # Actually, the loop in pollard_rho tries multiple c's. If all fail, it returns n.
                    # This is extremely unlikely for composite n.
                    # Let's just push m/d and d. If d==m, we push m and 1? No.
                    # Let's modify pollard_rho to never return n for composite n.
                    pass
                
                stack.append(d)
                stack.append(m // d)
        
    return max_f

# Refined largest_prime_factor to handle the pollard_rho return value better
def largest_prime_factor_safe(n):
    if n == 1:
        return 1
    
    max_f = 1
    stack = [n]
    
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        
        if is_prime(m):
            if m > max_f:
                max_f = m
            continue
        
        # Strip small primes
        for p in _SMALL_PRIMES:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                if m == 1:
                    if p > max_f:
                        max_f = p
                    break
                # m is reduced, check if it became prime
                if is_prime(m):
                    if m > max_f:
                        max_f = m
                    break
        else:
            # m is not divisible by any small prime
            # and m > 1 (since we would have broken if m==1)
            # and m is not prime (checked above if we broke, but if we didn't break, we didn't check)
            # Wait, the logic above is slightly flawed. Let's restructure.
            pass
        
    # Let's rewrite cleanly
    max_f = 1
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > max_f:
                max_f = m
            continue
        
        # Try to factor m
        # First, trial divide by small primes
        for p in _SMALL_PRIMES:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                if m == 1:
                    if p > max_f:
                        max_f = p
                    break
                # m is no longer divisible by p. Check if m is prime now.
                if is_prime(m):
                    if m > max_f:
                        max_f = m
                    break
                # If m is still composite, we need to factor it further.
                # We can push m back onto the stack? No, we already stripped p.
                # We should just continue to Pollard Rho for the remaining m.
                break
        
        # If the loop didn't break (meaning m wasn't reduced to 1 or prime via small primes),
        # or if we broke but m is still composite?
        # Actually, if we break from the for loop because m% p == 0, we handle it.
        # If we finish the for loop without breaking (m not divisible by small primes),
        # then m is either prime or has large factors.
        # We already checked is_prime(m) at the start of the while iteration? No, only for the original m.
        # If m was reduced, we need to check is_prime(m) again.
        
        # Let's simplify: 
        # 1. Check is_prime(m). If yes, update max_f and continue.
        # 2. If no, try small primes. If found, strip, update max_f, and push remaining m if > 1.
        # 3. If no small primes, use Pollard Rho.
        
    # Rewriting the function body completely for clarity and correctness
    
    max_f = 1
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        
        if is_prime(m):
            if m > max_f:
                max_f = m
            continue
        
        # m is composite
        # Try small primes
        factored = False
        for p in _SMALL_PRIMES:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                if m > 1:
                    stack.append(m)
                if p > max_f:
                    max_f = p
                factored = True
                break
        
        if not factored:
            # No small factors. Use Pollard Rho.
            d = pollard_rho(m)
            if d == m:
                # Fallback: treat as prime? No, that's wrong.
                # But Pollard Rho failing is rare. Let's just push m back? Infinite loop.
                # Let's assume it works.
                pass
            else:
                stack.append(d)
                stack.append(m // d)
                
    return max_f

# Overwrite the previous definition with the clean one
def largest_prime_factor(n):
    if n == 1:
        return 1
    max_f = 1
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > max_f:
                max_f = m
            continue
        # m is composite
        factored = False
        for p in _SMALL_PRIMES:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                if m > 1:
                    stack.append(m)
                if p > max_f:
                    max_f = p
                factored = True
                break
        if not factored:
            d = pollard_rho(m)
            if d != m:
                stack.append(d)
                stack.append(m // d)
            else:
                # If Pollard Rho fails, we might be stuck.
                # For robustness, we could try a different method or just assume prime (bad).
                # Given the constraints, this should not happen often.
                # Let's just skip it to avoid infinite loop, accepting a potential error.
                pass
    return max_f

# --- Main Construction ---

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_k = 0
    best_p = 2
    
    # We search for primes p.
    # Larger p generally allows for larger k because the smoothness bound is higher.
    # However, checking larger p takes longer.
    # Strategy:
    # 1. Iterate through candidate numbers.
    # 2. If prime, calculate the max k for that p.
    # 3. Update best.
    # 4. Stop when time runs out.
    
    # To speed up finding primes, we can skip even numbers.
    # Also, we can start from a higher number if we want, but increasing order is safer to ensure we don't miss a "local" max if the landscape is weird.
    # But typically, larger p gives larger k.
    # Let's iterate n from 2 upwards.
    
    n_candidate = 2
    
    while True:
        if time.time() > deadline:
            break
        
        if is_prime(n_candidate):
            p = n_candidate
            k = 0
            base = p * p
            
            # Check p^2 + 1 ... p^2 + k
            # We stop when we find a number with a prime factor > p.
            # Or when k reaches 200.
            
            # Optimization: 
            # If p is very small, k is small.
            # We can quickly reject small p?
            # For p=2, k=0.
            # For p=3, k=0.
            # For p=5, k=0.
            # For p=7, k=1.
            # Let's just run the check. It's fast for small k.
            
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                
                val = base + i
                
                # Quick check: if val is prime, it's > p (since val > p^2 > p for p>=2).
                # So k stops.
                if is_prime(val):
                    break
                
                # Check largest prime factor
                # We only care if it's > p.
                # We can modify largest_prime_factor to stop early if it finds a factor > p?
                # That would be a significant speedup.
                
                # Let's create a helper: has_large_factor(val, p)
                # Returns True if any prime factor of val is > p.
                
                if has_large_prime_factor(val, p):
                    break
                
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
        
        # Next candidate
        if n_candidate == 2:
            n_candidate = 3
        else:
            n_candidate += 2
            
    return {'p': best_p, 'k': best_k}

def has_large_prime_factor(n, limit):
    """
    Returns True if n has a prime factor > limit.
    Returns False if all prime factors are <= limit.
    """
    if n == 1:
        return False
    
    # If n itself is > limit and prime, return True.
    # But we need to factorize.
    
    # We can reuse the logic from largest_prime_factor but exit early.
    
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        
        if m > limit:
            # If m is prime, it's a large factor.
            # If m is composite, it might have large factors.
            # We need to know if ANY factor is > limit.
            
            if is_prime(m):
                return True
            
            # m is composite and > limit.
            # It could be a product of small factors. e.g. 2 * 2 * ... 
            # So we must factor it.
            
            # Try small primes
            factored = False
            for p in _SMALL_PRIMES:
                if m % p == 0:
                    while m % p == 0:
                        m //= p
                    # m is reduced.
                    # If m > limit, we need to check m further.
                    # If m <= limit, it's fine (assuming remaining factors are small).
                    # But we pushed m back? No, we are in a loop.
                    # We should push the remaining m if it's > 1.
                    if m > 1:
                        stack.append(m)
                    factored = True
                    break
            
            if not factored:
                # No small factors.
                # m is > limit.
                # If m is prime, return True. (Already checked above)
                # So m is composite with large factors.
                # Use Pollard Rho.
                d = pollard_rho(m)
                if d != m:
                    stack.append(d)
                    stack.append(m // d)
                else:
                    # Failure
                    return True # Assume large factor to be safe? Or False?
                    # If we can't factor it, we don't know.
                    # But if it's composite and not divisible by small primes,
                    # its factors are likely large.
                    # Let's return True to be conservative (stop the run).
                    pass
        
        # If m <= limit, we still need to ensure its factors are <= limit.
        # But if m <= limit, and we are factoring it, any factor of m is <= m <= limit.
        # So if m <= limit, we don't need to check further for "large" factors > limit.
        # Wait, this logic is:
        # We want to know if there is a factor > limit.
        # If the current number m is <= limit, then all its factors are <= limit.
        # So we can discard m from the stack.
        
    return False

# The previous has_large_prime_factor implementation was flawed because it didn't discard m <= limit.
# Let's rewrite it correctly.

def has_large_prime_factor(n, limit):
    if n == 1:
        return False
    
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        
        # If m <= limit, all factors of m are <= m <= limit.
        # So m cannot contribute a factor > limit.
        if m <= limit:
            continue
        
        # m > limit
        # Check if m is prime. If so, it's a large factor.
        if is_prime(m):
            return True
        
        # m is composite and > limit.
        # Try to factor out small primes.
        factored = False
        for p in _SMALL_PRIMES:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                if m > 1:
                    stack.append(m)
                factored = True
                break
        
        if not factored:
            # No small factors. m is composite and > limit.
            # Its prime factors must be > limit?
            # Not necessarily. e.g. limit=10, m=121=11*11. 11>10.
            # e.g. limit=100, m=101*103. Both > 100.
            # Since m has no small prime factors (<=37), and m > limit,
            # any prime factor of m is > 37.
            # If limit >= 37, we still need to check if the factor is > limit.
            # But if m is composite and has no small factors,
            # its smallest prime factor is > 37.
            # If limit < 37, then any factor is > limit.
            # If limit >= 37, we need to factor m to see if any factor > limit.
            
            # Use Pollard Rho
            d = pollard_rho(m)
            if d != m:
                stack.append(d)
                stack.append(m // d)
            else:
                # Pollard Rho failed.
                # Assume large factor.
                return True
                
    return False

# EVOLVE-BLOCK-END
