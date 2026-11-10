"""Stronger search for consecutive powerful-part product."""
import time
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i, v in enumerate(sieve) if v]

def _factorize(num, primes):
    """Return dict of prime -> exponent for num using primes up to sqrt(num)."""
    factors = {}
    temp = num
    for p in primes:
        if p * p > temp:
            break
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    return factors

def _powerful_part(num, primes):
    """Return the powerful part of num: product of p^e for e>=2."""
    factors = _factorize(num, primes)
    pp = 1
    for p, e in factors.items():
        if e >= 2:
            pp *= p ** e
    return pp

def _generate_powerful_cores(max_val):
    """
    Generate all powerful numbers (products of p^e, e>=2) up to max_val.
    Return list of (powerful_number, set_of_primes).
    """
    primes = _sieve_primes(int(max_val ** 0.5) + 10)
    # Actually we need primes up to max_val for the squarefree part check,
    # but for generating powerful cores, primes up to sqrt(max_val) suffice for e>=2.
    # Wait, p^2 <= max_val => p <= sqrt(max_val).
    
    cores = [(1, frozenset())]
    # Use a set to avoid duplicates? No, each powerful number has a unique factorization.
    # But we can generate recursively.
    
    # Simpler: iterate through all numbers? No, that's slow.
    # Generate by combining prime powers.
    
    prime_powers = []
    for p in primes:
        if p * p > max_val:
            break
        pp = p * p
        while pp <= max_val:
            prime_powers.append((pp, frozenset([p])))
            # Next power
            if pp > max_val // p:
                break
            pp *= p
    
    # Now combine these prime powers.
    # This is a knapsack-like problem or just simple multiplication.
    # Since max_val is 1e6, the number of powerful numbers is small (~1000s).
    
    # Let's just find all powerful numbers by checking each number?
    # 1e6 numbers, factoring each is O(sqrt(n)) ~ 1000 ops. 1e9 ops total. Too slow in Python.
    
    # Better: Generate powerful numbers directly.
    # A powerful number is of the form a^2 * b^3.
    # Let's generate all a^2 * b^3 <= max_val.
    
    powerful_numbers = set()
    max_a = int(math.isqrt(max_val))
    max_b = int(round(max_val ** (1/3)))
    
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > max_val:
            break
        max_a_for_b = int(math.isqrt(max_val // b3))
        for a in range(1, max_a_for_b + 1):
            val = a * a * b3
            if val <= max_val:
                powerful_numbers.add(val)
    
    # Now for each powerful number, compute its prime support and powerful part (which is itself).
    # We need the set of primes in the factorization.
    # We already have primes up to sqrt(max_val).
    # For val <= 1e6, sqrt(val) <= 1000.
    
    result = []
    primes_small = _sieve_primes(1000)
    
    for pn in powerful_numbers:
        factors = _factorize(pn, primes_small)
        support = frozenset(factors.keys())
        # Powerful part of pn is pn itself.
        result.append((pn, support))
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    max_n = 1000000
    min_n = 10000
    
    # Generate powerful cores
    cores = _generate_powerful_cores(max_n)
    # Sort by value descending to find high products faster?
    # Or just iterate.
    cores.sort(key=lambda x: x[0], reverse=True)
    
    best_score = 0
    best_n = min_n
    
    # Map from powerful number to its support for quick lookup?
    # We need to check disjointness of supports.
    
    # For each n in [min_n, max_n], compute powerful_part(n) and powerful_part(n+1).
    # This is O(N * sqrt(N)) which is too slow for N=1e6.
    
    # Alternative: Iterate over pairs of powerful cores (A, B) with disjoint supports.
    # n = A * s1
    # n+1 = B * s2
    # A * s1 + 1 = B * s2
    # B * s2 - A * s1 = 1
    
    # This is a Diophantine equation.
    # Since A and B are fixed, we can solve for s1, s2.
    # But s1, s2 are squarefree and coprime to A, B respectively?
    # Actually, s1 is squarefree and coprime to A (since A contains all p^e for e>=2).
    # Similarly for s2.
    
    # Let's try a different approach:
    # Precompute powerful_part for all numbers in range?
    # Memory: 1e6 integers, fine.
    # Time: Factoring 1e6 numbers.
    # Sieve smallest prime factor (SPF) is O(N log log N).
    # Then factoring each number is O(log N).
    
    def compute_powerful_parts(limit):
        spf = list(range(limit + 1))
        for i in range(2, int(limit ** 0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, limit + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        pp = [1] * (limit + 1)
        for i in range(2, limit + 1):
            if spf[i] == i:
                # i is prime
                continue
            # Factorize i using spf
            x = i
            factors = {}
            while x > 1:
                p = spf[x]
                e = 0
                while x % p == 0:
                    x //= p
                    e += 1
                factors[p] = e
            
            ppi = 1
            for p, e in factors.items():
                if e >= 2:
                    ppi *= p ** e
            pp[i] = ppi
            
            # Optimization: if i = j * k, can we derive pp[i] from pp[j]?
            # Not easily, because exponents merge.
            # But we can do better:
            # pp[i] = pp[i // p^e] * (p^e if e>=2 else 1) ... wait.
            # Let i = m * p^e where p does not divide m.
            # Then pp[i] = pp[m] * (p^e if e>=2 else 1).
            # This is correct.
            
        # Let's re-do with this optimization.
        pp = [1] * (limit + 1)
        for i in range(2, limit + 1):
            p = spf[i]
            if spf[i] == i:
                pp[i] = 1
            else:
                # Divide out all factors of p
                x = i
                e = 0
                while x % p == 0:
                    x //= p
                    e += 1
                if e >= 2:
                    pp[i] = pp[x] * (p ** e)
                else:
                    pp[i] = pp[x]
        
        return pp
    
    if time.time() > deadline - 5:
        return best_n
    
    pp = compute_powerful_parts(max_n + 1)
    
    for n in range(min_n, max_n + 1):
        score = pp[n] * pp[n+1]
        if score > best_score:
            best_score = score
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
