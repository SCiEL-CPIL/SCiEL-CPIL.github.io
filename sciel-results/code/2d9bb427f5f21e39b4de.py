import time
import numpy as np
from math import isqrt, gcd
from functools import reduce

def construct():
    t0 = time.time()
    DEADLINE = 85

    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000
    MINN = 10_000

    # 1. Sieve primes up to MAXN
    is_prime = bytearray(b'\x01') * (MAXN + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(MAXN) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, MAXN + 1) if is_prime[i]]
    
    if elapsed() > DEADLINE:
        return 10000

    # 2. Generate all squarefull numbers <= MAXN
    # A number is squarefull if for all p|n, p^2|n.
    # We generate them by combining prime powers p^k (k>=2).
    
    # First, list all prime powers p^k <= MAXN with k>=2
    prime_powers = []
    for p in primes:
        if p * p > MAXN:
            break
        power = p * p
        while power <= MAXN:
            prime_powers.append((p, power))
            power *= p

    # Generate all products of these prime powers, ensuring distinct primes.
    # We use a recursive approach or iterative expansion.
    # Since the number of primes involved is small, we can use a set.
    squarefulls = {1}
    
    # Sort prime_powers by prime to handle distinctness easily?
    # Actually, we just need to ensure we don't pick two powers of the same prime.
    # Group by prime.
    from collections import defaultdict
    powers_by_prime = defaultdict(list)
    for p, pp in prime_powers:
        powers_by_prime[p].append(pp)
        
    # Backtracking/DFS to generate combinations
    def gen_squarefull(idx, current_val):
        if idx >= len(powers_by_prime):
            if current_val <= MAXN:
                squarefulls.add(current_val)
            return
        
        p = list(powers_by_prime.keys())[idx]
        powers = powers_by_prime[p]
        
        # Option 1: Don't use any power of p
        gen_squarefull(idx + 1, current_val)
        
        # Option 2: Use one of the powers of p
        for pw in powers:
            new_val = current_val * pw
            if new_val > MAXN:
                break
            gen_squarefull(idx + 1, new_val)

    # Convert keys to list for indexing
    prime_keys = list(powers_by_prime.keys())
    
    # Optimization: Prune if current_val already > MAXN
    # The recursive function handles this.
    
    # Run generation
    gen_squarefull(0, 1)
    
    if elapsed() > DEADLINE:
        return 10000

    sf_list = list(squarefulls)
    sf_list.sort()
    
    # 3. Find best pair (A, B) from squarefulls
    # A | n, B | n+1
    # n = k*A
    # k*A = m*B - 1
    # k*A + 1 = m*B
    # A * k ≡ -1 (mod B)
    
    best_score = 0
    best_n = 10000

    # Precompute modular inverses? No, depends on B.
    
    # To speed up, we can iterate A, then B.
    # Since we want large A and B, we can iterate in reverse?
    # But we need to check all.
    
    # Optimization: If A*B is very large, the step AB is large, so few candidates.
    # If A*B is small, many candidates.
    
    for A in sf_list:
        if elapsed() > DEADLINE:
            break
        for B in sf_list:
            if elapsed() > DEADLINE:
                break
            
            g = gcd(A, B)
            if g != 1:
                continue
                
            # Solve A*k ≡ -1 (mod B)
            # Inverse of A mod B
            try:
                # Python 3.8+ has pow(a, -1, m)
                inv_A = pow(A, -1, B)
            except ValueError:
                continue # Should not happen if gcd=1
                
            # k ≡ -inv_A (mod B)
            k_min = (-inv_A) % B
            
            # n = A * k
            # We need n in [MINN, MAXN]
            # n = A * (k_min + t*B)
            # n_min = A * k_min
            # If n_min < MINN, find smallest t such that A*(k_min + t*B) >= MINN
            # If n_min > MAXN, no solution for this pair (since A>0, k_min>=0)
            
            n_candidate = A * k_min
            
            if n_candidate > MAXN:
                continue
                
            if n_candidate < MINN:
                # Find t
                # A * (k_min + t*B) >= MINN
                # k_min + t*B >= MINN / A
                # t*B >= MINN/A - k_min
                # t >= (MINN/A - k_min) / B
                required_k = (MINN + A - 1) // A # ceil(MINN/A)
                if required_k <= k_min:
                    t = 0
                else:
                    diff = required_k - k_min
                    t = (diff + B - 1) // B # ceil(diff/B)
                    
                n_candidate = A * (k_min + t * B)
                
            if n_candidate > MAXN:
                continue
                
            # Calculate P(n) and P(n+1)
            # We need a fast factorization for n and n+1
            # Since n <= 10^6, we can use the prime list
            
            def calc_powerful_part(m):
                res = 1
                temp = m
                for p in primes:
                    if p * p > temp:
                        break
                    if temp % p == 0:
                        e = 0
                        while temp % p == 0:
                            temp //= p
                            e += 1
                        if e >= 2:
                            res *= (p ** e)
                if temp > 1:
                    # temp is prime, exponent 1
                    pass
                return res

            p_n = calc_powerful_part(n_candidate)
            p_np1 = calc_powerful_part(n_candidate + 1)
            
            score = p_n * p_np1
            if score > best_score:
                best_score = score
                best_n = n_candidate

    return best_n
