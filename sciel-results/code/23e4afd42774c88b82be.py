"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Known valid solution with small max denominator
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Efficient divisor generation using prime factorization
    def get_divisors(n):
        """Generate all divisors of n using trial division"""
        if n <= 0:
            return []
        divisors = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
            i += 1
        return sorted(divisors)
    
    # Core: find all pairs (x, y) with x <= y, x > min_val, y <= max_val, 1/x + 1/y = r
    # Using (px - q)(py - q) = q^2 where r = p/q
    def find_pairs_for_remainder(r, min_val, max_val):
        """Find all pairs (x, y) with min_val < x < y <= max_val such that 1/x + 1/y = r"""
        pairs = []
        if r <= 0:
            return pairs
        
        p, q = r.numerator, r.denominator
        q_sq = q * q
        
        # For 1/x + 1/y = p/q:
        # (px - q)(py - q) = q^2
        # Let d = px - q, then d | q^2, x = (d+q)/p, y = (q^2/d + q)/p
        # Need: d > 0, (d+q) % p == 0, x > min_val, x < y, y <= max_val
        
        # To ensure x < y: d < q^2/d => d^2 < q^2 => d < q
        # Also x > min_val => (d+q)/p > min_val => d > p*min_val - q
        # And y <= max_val => (q^2/d + q)/p <= max_val => q^2/d <= p*max_val - q => d >= q^2/(p*max_val - q)
        
        if q_sq <= 10**10:
            # Get divisors of q^2
            divs = get_divisors(q_sq)
            
            for d in divs:
                # d must be positive (it is since divisors of positive number)
                if d <= 0:
                    continue
                
                # x = (d + q) / p must be integer
                if (d + q) % p != 0:
                    continue
                x = (d + q) // p
                
                if x <= min_val or x > max_val:
                    continue
                
                # y = (q_sq // d + q) / p must be integer
                d2 = q_sq // d
                if (d2 + q) % p != 0:
                    continue
                y = (d2 + q) // p
                
                if y <= x or y > max_val:
                    continue
                
                pairs.append((x, y))
        else:
            # For very large q, iterate over x in valid range
            # x must satisfy: p*x - q > 0 and p*x - q divides x*q
            # x > q/p, and from y > x: q^2/(px-q) > px-q => (px-q)^2 < q^2 => px-q < q => x < 2q/p
            min_x = max(min_val + 1, q // p + 1)
            max_x = min(max_val, (2 * q) // p)
            
            for x in range(min_x, max_x + 1):
                denom = p * x - q
                if denom <= 0:
                    continue
                if (x * q) % denom != 0:
                    continue
                y = (x * q) // denom
                if y <= x or y > max_val:
                    continue
                pairs.append((x, y))
        
        return pairs
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = target
    def search_3_terms(max_denom):
        nonlocal best_solution, best_max
        
        for a in range(2, min(max_denom, 200) + 1):
            if time.time() > deadline:
                break
            
            r = target - Fraction(1, a)
            if r <= 0:
                continue
            
            pairs = find_pairs_for_remainder(r, a, max_denom)
            for b, c in pairs:
                if b > a and c > b and c <= max_denom:
                    candidate = [a, b, c]
                    if c < best_max:
                        best_max = c
                        best_solution = candidate
                        if best_max <= 15:
                            return
    
    # Search for 4-term solutions
    def search_4_terms(max_denom):
        nonlocal best_solution, best_max
        
        for a in range(2, min(max_denom, 80) + 1):
            if time.time() > deadline:
                break
            
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            
            for b in range(a + 1, min(max_denom, 150) + 1):
                if time.time() > deadline:
                    break
                
                r = r_after_a - Fraction(1, b)
                if r <= 0:
                    continue
                
                pairs = find_pairs_for_remainder(r, b, max_denom)
                for c, d in pairs:
                    if c > b and d > c and d <= max_denom:
                        candidate = [a, b, c, d]
                        if d < best_max:
                            best_max = d
                            best_solution = candidate
                            if best_max <= 15:
                                return
    
    # Search for 5-term solutions
    def search_5_terms(max_denom):
        nonlocal best_solution, best_max
        
        for a in range(2, min(max_denom, 50) + 1):
            if time.time() > deadline:
                break
            
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            
            for b in range(a + 1, min(max_denom, 100) + 1):
                if time.time() > deadline:
                    break
                
                r_after_b = r_after_a - Fraction(1, b)
                if r_after_b <= 0:
                    continue
                
                for c in range(b + 1, min(max_denom, 150) + 1):
                    if time.time() > deadline:
                        break
                    
                    r = r_after_b - Fraction(1, c)
                    if r <= 0:
                        continue
                    
                    pairs = find_pairs_for_remainder(r, c, max_denom)
                    for d, e in pairs:
                        if d > c and e > d and e <= max_denom:
                            candidate = [a, b, c, d, e]
                            if e < best_max:
                                best_max = e
                                best_solution = candidate
                                if best_max <= 15:
                                    return
    
    # Search for 6-term solutions (for very small max_denom)
    def search_6_terms(max_denom):
        nonlocal best_solution, best_max
        
        for a in range(2, min(max_denom, 35) + 1):
            if time.time() > deadline:
                break
            
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            
            for b in range(a + 1, min(max_denom, 60) + 1):
                if time.time() > deadline:
                    break
                
                r_after_b = r_after_a - Fraction(1, b)
                if r_after_b <= 0:
                    continue
                
                for c in range(b + 1, min(max_denom, 100) + 1):
                    if time.time() > deadline:
                        break
                    
                    r_after_c = r_after_b - Fraction(1, c)
                    if r_after_c <= 0:
                        continue
                    
                    for d in range(c + 1, min(max_denom, 150) + 1):
                        if time.time() > deadline:
                            break
                        
                        r = r_after_c - Fraction(1, d)
                        if r <= 0:
                            continue
                        
                        pairs = find_pairs_for_remainder(r, d, max_denom)
                        for e, f in pairs:
                            if e > d and f > e and f <= max_denom:
                                candidate = [a, b, c, d, e, f]
                                if f < best_max:
                                    best_max = f
                                    best_solution = candidate
                                    if best_max <= 15:
                                        return
    
    # Systematic search with increasing bounds
    bounds = [15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000]
    
    for bound in bounds:
        if time.time() > deadline:
            break
        
        # Try 3-term
        search_3_terms(bound)
        if best_max <= bound:
            continue
        
        # Try 4-term
        search_4_terms(bound)
        if best_max <= bound:
            continue
        
        # Try 5-term
        search_5_terms(bound)
        if best_max <= bound:
            continue
        
        # Try 6-term for small bounds
        if bound <= 100:
            search_6_terms(bound)
    
    # Final verification and fallback to known valid solution
    def is_valid_solution(sol):
        if not sol:
            return False
        if len(sol) > 256:
            return False
        if len(set(sol)) != len(sol):
            return False
        if any(d < 1 or d > 10000000 for d in sol):
            return False
        s = sum(Fraction(1, d) for d in sol)
        return s == target

    if not is_valid_solution(best_solution):
        best_solution = [8, 31, 248]
        best_max = 248
    
    return best_solution
# EVOLVE-BLOCK-END
