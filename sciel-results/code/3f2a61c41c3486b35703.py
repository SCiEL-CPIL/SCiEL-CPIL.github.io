"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Greedy fallback solution
    def greedy_solution(target):
        denominators = []
        r = target
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            denominators.append(d)
            r -= Fraction(1, d)
        return denominators
    
    best_solution = greedy_solution(target)
    best_max = max(best_solution)
    
    # Optimized divisor generation using prime factorization
    def get_divisors(n):
        if n == 1:
            return [1]
        divisors = [1]
        temp_n = n
        p = 2
        while p * p <= temp_n:
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                # Multiply existing divisors by p^k
                current_divs = list(divisors)
                mult = 1
                for _ in range(count):
                    mult *= p
                    for d in current_divs:
                        divisors.append(d * mult)
            p += 1
        if temp_n > 1:
            current_divs = list(divisors)
            for d in current_divs:
                divisors.append(d * temp_n)
        return sorted(divisors)

    def find_pairs_for_remainder(r, min_val, max_val):
        """Find all pairs (x, y) with min_val < x < y <= max_val such that 1/x + 1/y = r"""
        pairs = []
        if r <= 0:
            return pairs
        
        p, q = r.numerator, r.denominator
        q_sq = q * q
        
        # Generate divisors of q^2
        divisors = get_divisors(q_sq)
        
        for d in divisors:
            # x = (d + q) / p
            if (d + q) % p != 0:
                continue
            x = (d + q) // p
            if x <= min_val or x > max_val:
                continue
            
            # y = (q_sq // d + q) / p
            d2 = q_sq // d
            if (d2 + q) % p != 0:
                continue
            y = (d2 + q) // p
            if y <= x or y > max_val:
                continue
            
            # Verify
            if Fraction(1, x) + Fraction(1, y) != r:
                continue
            
            pairs.append((x, y))
        
        return pairs

    # Search for 3-term solutions
    def search_3_terms(max_denom):
        nonlocal best_solution, best_max
        
        # a must be > q/p
        min_a = q // p + 1
        # 1/a + 1/b + 1/c = 5/31
        # 1/a < 5/31 => a > 6.2 => a >= 7
        # 3/a > 5/31 => a < 18.6 => a <= 18
        
        for a in range(max(min_a, 2), min(max_denom, 20) + 1):
            if time.time() > deadline:
                break
            
            r = target - Fraction(1, a)
            if r <= 0:
                continue
            
            # Find pairs (b, c) with a < b < c <= max_denom
            pairs = find_pairs_for_remainder(r, a, max_denom)
            for b, c in pairs:
                if b > a and c > b and c <= max_denom:
                    candidate = [a, b, c]
                    if c < best_max:
                        best_max = c
                        best_solution = candidate

    # Search for 4-term solutions
    def search_4_terms(max_denom):
        nonlocal best_solution, best_max
        
        # a < b < c < d
        # a >= 7 (from 3-term analysis, but with 4 terms, a can be smaller?
        # 1/a < 5/31 => a > 6.2 => a >= 7)
        # 4/a > 5/31 => a < 12.4 => a <= 12
        
        for a in range(7, min(max_denom, 15) + 1):
            if time.time() > deadline:
                break
            
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            
            # b must be such that 1/b < r_after_a
            # 3/b > r_after_a => b < 3/r_after_a
            max_b = min(max_denom, int(3 / r_after_a) + 1)
            
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline:
                    break
                
                r = r_after_a - Fraction(1, b)
                if r <= 0:
                    continue
                
                # Find pairs (c, d) with b < c < d <= max_denom
                pairs = find_pairs_for_remainder(r, b, max_denom)
                for c, d in pairs:
                    if c > b and d > c and d <= max_denom:
                        candidate = [a, b, c, d]
                        if d < best_max:
                            best_max = d
                            best_solution = candidate

    # Search for 5-term solutions
    def search_5_terms(max_denom):
        nonlocal best_solution, best_max
        
        # a >= 7
        # 5/a > 5/31 => a < 31 => a <= 30
        
        for a in range(7, min(max_denom, 35) + 1):
            if time.time() > deadline:
                break
            
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            
            max_b = min(max_denom, int(4 / r_after_a) + 1)
            for b in range(a + 1, max_b + 1):
                if time.time() > deadline:
                    break
                
                r_after_b = r_after_a - Fraction(1, b)
                if r_after_b <= 0:
                    continue
                
                max_c = min(max_denom, int(3 / r_after_b) + 1)
                for c in range(b + 1, max_c + 1):
                    if time.time() > deadline:
                        break
                    
                    r = r_after_b - Fraction(1, c)
                    if r <= 0:
                        continue
                    
                    # Find pairs (d, e) with c < d < e <= max_denom
                    pairs = find_pairs_for_remainder(r, c, max_denom)
                    for d, e in pairs:
                        if d > c and e > d and e <= max_denom:
                            candidate = [a, b, c, d, e]
                            if e < best_max:
                                best_max = e
                                best_solution = candidate

    # Systematic search
    # Start with small bounds and increase
    # The optimal max denominator is likely in the range 100-2000
    # We will search for 3, 4, 5 terms with increasing max_denom bounds
    
    bounds = [100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000, 1000000]
    
    for bound in bounds:
        if time.time() > deadline:
            break
        
        # Try 3-term
        search_3_terms(bound)
        
        # Try 4-term
        if best_max > bound:
            search_4_terms(bound)
            
        # Try 5-term
        if best_max > bound:
            search_5_terms(bound)
        
        # Early termination if we found a very small max
        # Theoretical minimum for 5/31:
        # If max is M, and k terms, sum <= k/M.
        # 5/31 <= k/M => M <= 31k/5 = 6.2k.
        # For k=3, M <= 18.6. But we found no 3-term solution with small M.
        # For k=4, M <= 24.8.
        # For k=5, M <= 31.
        # We found [7, 56, 1736] with M=1736.
        # Let's see if we can find smaller.
        
        if best_max <= 100:
            break

    # Final verification
    if best_solution is not None:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            # Fallback to greedy
            best_solution = greedy_solution(target)
            best_max = max(best_solution)
        
        # Ensure constraints
        if len(best_solution) > 256 or len(set(best_solution)) != len(best_solution) or any(d < 1 or d > 10000000 for d in best_solution):
            best_solution = greedy_solution(target)
            best_max = max(best_solution)
    
    return best_solution
# EVOLVE-BLOCK-END
