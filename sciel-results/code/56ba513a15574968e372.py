"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt
from collections import defaultdict

def _factorize(n):
    """Return prime factorization of n as dict {prime: exponent}."""
    if n <= 1:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factorization(factors):
    """Generate all divisors from prime factorization."""
    if not factors:
        return [1]
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        for d in divisors:
            for e in range(exponents[i] + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    return sorted(divisors)

def _get_factor_pairs(n):
    """Return all (u, v) pairs with u*v = n, u <= v."""
    if n <= 1:
        return [(1, n)] if n == 1 else []
    factors = _factorize(n)
    divs = _divisors_from_factorization(factors)
    pairs = []
    for u in divs:
        if u * u > n:
            break
        if n % u == 0:
            v = n // u
            pairs.append((u, v))
    return pairs

def construct():
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Start with known solution
    best_solution = [8, 31, 248]
    best_max = 248
    
    # 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c
    # For each a, rem = 5/31 - 1/a = p/q
    # 1/b + 1/c = p/q => (p*b - q)(p*c - q) = q^2
    # Enumerate factor pairs of q^2 efficiently
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        p = rem.numerator
        q = rem.denominator
        q2 = q * q
        
        # Get factor pairs of q^2
        pairs = _get_factor_pairs(q2)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            # b = (u + q) / p, c = (v + q) / p
            if (u + q) % p != 0:
                continue
            b = (u + q) // p
            if b <= a:
                continue
            if (v + q) % p != 0:
                continue
            c = (v + q) // p
            if c <= b:
                continue
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
    
    # 4-term search: 1/a + 1/b + 1/c + 1/d = 5/31
    for a in range(7, 25):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = max(a + 1, int(1 / rem_a) + 1)
        b_max = min(int(3 / rem_a), best_max - 1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_bc = rem_a - Fraction(1, b)
            if rem_bc <= 0:
                break
            p = rem_bc.numerator
            q = rem_bc.denominator
            q2 = q * q
            c_min = max(b + 1, int(1 / rem_bc) + 1)
            
            pairs = _get_factor_pairs(q2)
            for u, v in pairs:
                if time.time() > deadline:
                    break
                if (u + q) % p != 0:
                    continue
                c = (u + q) // p
                if c < c_min or c <= b:
                    continue
                if (v + q) % p != 0:
                    continue
                d = (v + q) // p
                if d <= c:
                    continue
                if d < best_max:
                    best_max = d
                    best_solution = [a, b, c, d]
    
    # 5-term search
    for a in range(7, 31):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = max(a + 1, int(1 / rem_a) + 1)
        b_max = min(int(4 / rem_a), best_max - 1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            c_min = max(b + 1, int(1 / rem_b) + 1)
            c_max = min(int(3 / rem_b), best_max - 1)
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                p = rem_c.numerator
                q = rem_c.denominator
                q2 = q * q
                d_min = max(c + 1, int(1 / rem_c) + 1)
                
                pairs = _get_factor_pairs(q2)
                for u, v in pairs:
                    if time.time() > deadline:
                        break
                    if (u + q) % p != 0:
                        continue
                    d = (u + q) // p
                    if d < d_min or d <= c:
                        continue
                    if (v + q) % p != 0:
                        continue
                    e = (v + q) // p
                    if e <= d:
                        continue
                    if e < best_max:
                        best_max = e
                        best_solution = [a, b, c, d, e]
    
    # 6-term search
    for a in range(7, 38):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = max(a + 1, int(1 / rem_a) + 1)
        b_max = min(int(5 / rem_a), best_max - 1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            c_min = max(b + 1, int(1 / rem_b) + 1)
            c_max = min(int(4 / rem_b), best_max - 1)
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                d_min = max(c + 1, int(1 / rem_c) + 1)
                d_max = min(int(3 / rem_c), best_max - 1)
                
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        break
                    p = rem_d.numerator
                    q = rem_d.denominator
                    q2 = q * q
                    e_min = max(d + 1, int(1 / rem_d) + 1)
                    
                    pairs = _get_factor_pairs(q2)
                    for u, v in pairs:
                        if time.time() > deadline:
                            break
                        if (u + q) % p != 0:
                            continue
                        e = (u + q) // p
                        if e < e_min or e <= d:
                            continue
                        if (v + q) % p != 0:
                            continue
                        f = (v + q) // p
                        if f <= e:
                            continue
                        if f < best_max:
                            best_max = f
                            best_solution = [a, b, c, d, e, f]
    
    # Final verification
    s = sum(Fraction(1, d) for d in best_solution)
    assert s == target, f"Sum {s} != {target}"
    assert len(set(best_solution)) == len(best_solution), "Duplicate denominators"
    assert all(1 <= d <= 10000000 for d in best_solution), "Denominator out of range"
    assert 1 <= len(best_solution) <= 256, "Invalid term count"
    
    return best_solution
# EVOLVE-BLOCK-END
