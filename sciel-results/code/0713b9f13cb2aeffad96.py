"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # Get primes
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # Start with primes + 1
    current_set = primes + [1]
    
    # Helper: check if a set is valid (pairwise coprime)
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    # Helper: get prime factors of a number
    def prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    # For each prime p in current set, try to replace it with p*k
    # where k is a product of other primes (or 1) such that p*k <= 256
    # and the new set remains valid
    
    best_sum = sum(current_set)
    best_set = list(current_set)
    
    # Local search: iteratively try to improve
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n = len(current_set)
        
        for i in range(n):
            if time.time() > deadline - 5:
                break
            p = current_set[i]
            if p == 1:
                continue
            
            # Try replacing p with p*k for various k
            # k must be coprime to all other elements in the set (except p)
            # and p*k <= MAX_N
            
            # Get the other elements
            others = [current_set[j] for j in range(n) if j != i]
            
            # Try k from 2 to MAX_N//p
            for k in range(2, MAX_N // p + 1):
                new_val = p * k
                if new_val > MAX_N:
                    break
                
                # Check if new_val is coprime to all others
                valid = True
                for o in others:
                    if gcd(new_val, o) != 1:
                        valid = False
                        break
                
                if valid:
                    # Also need new_val not already in the set
                    if new_val not in others:
                        gain = new_val - p
                        if gain > 0:
                            # Try this replacement
                            new_set = list(current_set)
                            new_set[i] = new_val
                            new_sum = sum(new_set)
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_set = new_set
                                current_set = new_set
                                improved = True
                                break  # Restart search with new set
    
    # Final safety check
    if not is_valid(best_set):
        best_set = primes + [1]
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
