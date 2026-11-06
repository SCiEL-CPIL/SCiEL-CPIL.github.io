"""Find N <= 10^18 with many representations as sum of two powerful numbers."""
import time
import random
from math import isqrt, log
import sys

def _get_primes_1_mod_4(limit):
    """Generate primes <= limit that are 1 mod 4."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(5, limit + 1) if sieve[i] and i % 4 == 1]

def _find_best_K(Q, limit_N=10**18):
    """
    Find K such that N = Q*K <= limit_N maximizes the number of representations
    of K as a sum of two squares.
    The number of representations is proportional to prod(e_i + 1) where K = prod p_i^e_i
    and p_i are primes 1 mod 4.
    """
    max_K = limit_N // Q
    if max_K < 1:
        return 1, 1
    
    # We need primes p_i = 1 mod 4.
    # Let's get enough primes. 5^20 is ~ 9.5e13. 5^25 is ~ 2.9e17.
    # So we need primes up to maybe 1000 to be safe for higher exponents of smaller primes.
    primes = _get_primes_1_mod_4(1000)
    
    best_score = 0
    best_K = 1
    
    # Recursive search to maximize prod(e_i + 1)
    # We assign exponents e_i to primes p_i.
    # Constraint: prod(p_i^e_i) <= max_K
    # Objective: maximize prod(e_i + 1)
    
    def search(idx, current_K, current_score):
        nonlocal best_score, best_K
        
        # Pruning: even if we set all remaining exponents to 1, can we beat best_score?
        # This is hard to estimate tightly, so we rely on the small depth.
        
        if idx >= len(primes):
            if current_score > best_score:
                best_score = current_score
                best_K = current_K
            return
            
        p = primes[idx]
        
        # Try exponents e = 1, 2, ...
        # We can also try e = 0 (skip this prime)
        # But usually using smaller primes with higher exponents is better for the product?
        # Actually, for a fixed product K, distributing prime factors among many distinct primes
        # yields a higher product (e_i+1) than concentrating them.
        # Example: 5^2 = 25. Score (2+1)=3.
        # 5 * 13 = 65. Score (1+1)(1+1)=4.
        # So we should try to use as many distinct primes as possible, with small exponents.
        
        # Let's iterate e from 1 upwards.
        # Also consider e=0 to move to next prime.
        
        # Optimization: If current_K * p > max_K, we can't use this prime or higher.
        if current_K * p > max_K:
            # We can't use p^1, so we can't use any higher power.
            # We must skip this prime and all subsequent larger primes.
            # Just update best if current state is better.
            if current_score > best_score:
                best_score = current_score
                best_K = current_K
            return

        # Try e=1, 2, ...
        power = p
        e = 1
        while current_K * power <= max_K:
            # New state
            new_K = current_K * power
            new_score = current_score * (e + 1)
            
            # Recurse to next prime
            search(idx + 1, new_K, new_score)
            
            # Increase exponent
            power *= p
            e += 1
            
        # Option: Skip this prime (e=0)
        # This is implicitly handled by not calling search with e>=1?
        # No, we need to explicitly try skipping.
        # Wait, the loop above tries e=1, 2...
        # We should also try e=0.
        # But if we skip p, we might use larger primes.
        search(idx + 1, current_K, current_score)

    search(0, 1, 1)
    
    return best_K, best_score

def _find_sum_of_squares_pairs(K):
    """Find all pairs (u, v) with u <= v, u^2 + v^2 = K."""
    pairs = []
    # u^2 <= K/2
    max_u = isqrt(K // 2)
    for u in range(0, max_u + 1):
        v2 = K - u * u
        v = isqrt(v2)
        if v * v == v2 and u <= v:
            pairs.append((u, v))
    return pairs

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Try different square-free parts Q
    # Q=1 is the simplest.
    # Q=2, Q=3, Q=6, Q=30, etc.
    # Larger Q reduces the space for K, so fewer representations.
    # Q=1 is likely best.
    
    Qs = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 91, 93, 94, 95, 97]
    
    for Q in Qs:
        if time.time() > deadline:
            break
            
        K, score = _find_best_K(Q)
        N = Q * K
        
        if N > 10**18:
            continue
            
        # Find pairs (u, v) for K
        uv_pairs = _find_sum_of_squares_pairs(K)
        
        # Construct powerful number pairs
        # a = Q * u^2, b = Q * v^2
        # Check if a, b are powerful.
        # u^2 is a square, so Q*u^2 is powerful iff Q is powerful?
        # No. Powerful number definition: every prime factor appears squared.
        # If Q is square-free, Q*u^2 is powerful iff Q=1.
        # Wait.
        # Let Q = prod p_i.
        # a = Q * u^2.
        # Prime factorization of a: p_i appears with exponent 1 + 2*e_{i,u}.
        # For a to be powerful, all exponents must be >= 2 and even? No, just >= 2?
        # "every prime factor appears squared" usually means if p|n, then p^2|n.
        # So exponents must be >= 2.
        # If Q is square-free and Q > 1, then p_i appears with exponent 1 + 2*k.
        # If k=0, exponent is 1. Not powerful.
        # If k>=1, exponent is >= 3. Powerful.
        # So u must be divisible by all primes in Q.
        # Let u = Q * x.
        # Then a = Q * (Q*x)^2 = Q^3 * x^2.
        # Exponent of p_i in a is 3 + 2*e_{i,x} >= 3. Powerful.
        # So we need u to be a multiple of Q.
        
        # Let's re-evaluate.
        # We want a, b powerful.
        # Let a = Q * u^2.
        # If Q is square-free, for a to be powerful, u must be divisible by Q.
        # Let u = Q * x.
        # Then a = Q * Q^2 * x^2 = Q^3 * x^2.
        # Similarly b = Q * v^2 with v = Q * y.
        # b = Q^3 * y^2.
        # Then a + b = Q^3 (x^2 + y^2).
        # So N = Q^3 * K'.
        # This changes the construction.
        
        # Let's stick to the simplest case: Q=1.
        # Then a = u^2, b = v^2.
        # Squares are powerful.
        # So N = u^2 + v^2.
        # This is the standard "sum of two squares" problem.
        # We want N <= 10^18 with many representations as sum of two squares.
        
        # My previous logic for _find_best_K assumed K is the number to be represented as sum of two squares.
        # If Q=1, N = K.
        # We maximize representations of K as sum of two squares.
        
        # Let's just use Q=1 for now, as it's the most efficient.
        if Q != 1:
             # We can try to adapt, but Q=1 is strong.
             # Let's just skip non-1 Q for simplicity and speed, or handle the u=Q*x constraint.
             # If we use u=Q*x, then N = Q^3 (x^2+y^2).
             # Let M = Q^3.
             # We want M * (x^2+y^2) <= 10^18.
             # So x^2+y^2 <= 10^18 / M.
             # This is the same problem with a smaller limit.
             # So we can call _find_best_K with limit 10^18 // (Q**3) and then scale?
             # No, the structure is different.
             # Let's just stick to Q=1.
             continue

        # For Q=1, we have N = K.
        # Pairs are (u^2, v^2).
        # We need to verify they are powerful. Squares are powerful.
        
        pw_pairs = []
        for u, v in uv_pairs:
            a = u * u
            b = v * v
            # Ensure a <= b
            if a <= b:
                pw_pairs.append([a, b])
        
        # Remove duplicates? The pairs (u,v) from _find_sum_of_squares_pairs are distinct and u<=v.
        # So (u^2, v^2) are distinct.
        
        if len(pw_pairs) > best_count:
            best_count = len(pw_pairs)
            best_N = N
            best_pairs = pw_pairs
            
    return {'N': best_N, 'pairs': best_pairs}
