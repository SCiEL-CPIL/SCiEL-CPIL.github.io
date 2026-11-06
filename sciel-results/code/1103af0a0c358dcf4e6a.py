"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def _sieve_factorizations(limit):
    """Precompute prime factorizations for all numbers up to limit.
    Returns a list where factorizations[i] is a dict of prime->exponent."""
    factorizations = [{} for _ in range(limit + 1)]
    factorizations[1] = {}
    # Sieve: for each prime p, divide all multiples
    for p in range(2, limit + 1):
        if factorizations[p]:  # already factored, not prime
            continue
        # p is prime
        for m in range(p, limit + 1, p):
            exp = 0
            while m % p == 0:
                exp += 1
                m //= p
            factorizations[m][p] = factorizations[m].get(p, 0) + exp
    return factorizations

def _merge_factors(f1, f2):
    """Merge two factorization dicts by adding exponents."""
    merged = dict(f1)
    for p, e in f2.items():
        merged[p] = merged.get(p, 0) + e
    return merged

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _find_solution_for_n(n, precomputed_factors, deadline):
    """Find distinct positive integer denominators x,y,z with 1/x+1/y+1/z=4/n."""
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    # Try x values in order of increasing B = n*x (smaller B means fewer divisors to check)
    # Actually, let's try x values that are likely to yield solutions quickly.
    # For each x, we need divisors d of B^2 where d < B and d ≡ -B (mod A)
    # Smaller B -> fewer divisors -> faster
    
    for x in range(x_min, x_max + 1):
        if time.time() > deadline:
            return None
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        
        # Get factorization of B^2
        # B = n * x, so factorization of B is merge of factorization of n and x
        # Then B^2 has doubled exponents
        if B > len(precomputed_factors) - 1:
            continue
        fB = precomputed_factors[B]
        fB2 = {p: 2 * e for p, e in fB.items()}
        
        B2 = B * B
        
        # We need divisors d of B^2 such that:
        # 1. d < B
        # 2. (d + B) % A == 0, i.e., d ≡ -B (mod A)
        # 3. y = (d + B) // A > x
        # 4. z = (B^2 // d + B) // A > y
        # 5. z <= 10^12
        
        # Generate divisors of B^2
        divs = _divisors_from_factors(fB2)
        
        # Filter: d < B and d ≡ -B (mod A)
        # -B mod A = (A - B % A) % A
        target_mod = (-B) % A
        
        for d in divs:
            if time.time() > deadline:
                return None
            if d >= B:
                continue
            if d % A != target_mod:
                continue
            num_y = d + B
            y = num_y // A
            if y <= x:
                continue
            # Check z
            v = B2 // d
            num_z = v + B
            if num_z % A != 0:
                continue
            z = num_z // A
            if z <= y:
                continue
            if z > 10**12:
                continue
            # Verify with cross multiplication
            lhs = n * (y * z + x * z + x * y)
            rhs = 4 * x * y * z
            if lhs == rhs:
                return [x, y, z]
    
    return None

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    
    # Precompute factorizations up to max possible B = 256 * 192 = 49152
    max_B = 256 * 192 + 10  # small buffer
    precomputed_factors = _sieve_factorizations(max_B)
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        results[idx] = _find_solution_for_n(n, precomputed_factors, deadline)
    
    return results
# EVOLVE-BLOCK-END
