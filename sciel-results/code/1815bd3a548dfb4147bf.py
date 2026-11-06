"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s

    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')

    # Phase 1: Exhaustive search for 2-term solutions
    # 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # (5a - 31)(5b - 31) = 31^2 = 961
    # So 5a - 31 must be a divisor of 961
    # 961 = 31^2, divisors: 1, 31, 961
    # 5a - 31 = d => a = (d + 31) / 5
    for d in [1, 31, 961]:
        num = d + 31
        if num % 5 == 0:
            a = num // 5
            if a >= 1:
                b_num = (961 // d) + 31
                if b_num % 5 == 0:
                    b = b_num // 5
                    if a != b and a >= 1 and b >= 1 and a <= 10000000 and b <= 10000000:
                        solution = [a, b]
                        max_denom = max(solution)
                        if max_denom < best_max_denom:
                            best_max_denom = max_denom
                            best_solution = solution

    # Phase 2: Exhaustive search for 3-term solutions using algebraic completion
    # 1/a + 1/b + 1/c = 5/31
    # For fixed a, we need 1/b + 1/c = 5/31 - 1/a = (5a - 31) / (31a)
    # Let r = (5a - 31) / (31a). We need 1/b + 1/c = r
    # (r.denominator) * (b+c) = r.numerator * b * c
    # Let r = p/q in lowest terms. Then q(b+c) = p*b*c
    # p*b*c - q*b - q*c = 0
    # (p*b - q)(p*c - q) = q^2
    # So p*b - q must be a divisor of q^2
    # b = (divisor + q) / p, c = (q^2/divisor + q) / p
    
    # We search a from 7 upward. For each a, compute r, then find all (b,c) pairs.
    # To minimize the max denominator, we want small a, b, c.
    
    a_start = 7
    a_end = 100  # Start with a small range; expand if needed
    
    for a in range(a_start, a_end + 1):
        if time.time() > deadline:
            break
        
        # r = 5/31 - 1/a = (5a - 31) / (31a)
        p = 5 * a - 31
        q = 31 * a
        
        if p <= 0:
            continue
            
        # Reduce r = p/q
        g = gcd(p, q)
        p_reduced = p // g
        q_reduced = q // g
        
        # Need (p_reduced * b - q_reduced)(p_reduced * c - q_reduced) = q_reduced^2
        # Let D = q_reduced^2
        # For each divisor d of D:
        #   b = (d + q_reduced) / p_reduced
        #   c = (D/d + q_reduced) / p_reduced
        
        D = q_reduced * q_reduced
        
        # Find all divisors of D
        # D can be large, but q_reduced = 31a/g where g = gcd(5a-31, 31a)
        # For small a, D is manageable
        if D > 10**12:
            continue
            
        divisors = []
        i = 1
        while i * i <= D:
            if D % i == 0:
                divisors.append(i)
                if i != D // i:
                    divisors.append(D // i)
            i += 1
        
        for d in divisors:
            # b = (d + q_reduced) / p_reduced
            num_b = d + q_reduced
            if num_b % p_reduced != 0:
                continue
            b = num_b // p_reduced
            
            if b <= a or b < 1 or b > 10000000:
                continue
            
            num_c = D // d + q_reduced
            if num_c % p_reduced != 0:
                continue
            c = num_c // p_reduced
            
            if c <= a or c < 1 or c > 10000000:
                continue
            
            if b == c:
                continue
            
            solution = [a, b, c]
            max_denom = max(solution)
            if max_denom < best_max_denom:
                best_max_denom = max_denom
                best_solution = solution

    # Phase 3: If no 3-term solution found, try 4-term with fixed a,b and algebraic completion
    if best_solution is None or best_max_denom > 200:
        # 1/a + 1/b + 1/c + 1/d = 5/31
        # Fix a, b. Then 1/c + 1/d = 5/31 - 1/a - 1/b
        # Use the same algebraic method for the remaining 2 terms
        
        for a in range(7, 50):
            if time.time() > deadline:
                break
            
            for b in range(a + 1, a + 200):
                if time.time() > deadline:
                    break
                
                # r = 5/31 - 1/a - 1/b = (5ab - 31b - 31a) / (31ab)
                p = 5 * a * b - 31 * b - 31 * a
                q = 31 * a * b
                
                if p <= 0:
                    continue
                
                g = gcd(p, q)
                p_reduced = p // g
                q_reduced = q // g
                
                D = q_reduced * q_reduced
                if D > 10**14:
                    continue
                
                # Find divisors of D
                divisors = []
                i = 1
                while i * i <= D:
                    if D % i == 0:
                        divisors.append(i)
                        if i != D // i:
                            divisors.append(D // i)
                    i += 1
                
                for d in divisors:
                    num_c = d + q_reduced
                    if num_c % p_reduced != 0:
                        continue
                    c = num_c // p_reduced
                    
                    if c <= b or c < 1 or c > 10000000:
                        continue
                    
                    num_dd = D // d + q_reduced
                    if num_dd % p_reduced != 0:
                        continue
                    dd = num_dd // p_reduced
                    
                    if dd <= b or dd < 1 or dd > 10000000:
                        continue
                    
                    if c == dd:
                        continue
                    
                    solution = [a, b, c, dd]
                    max_denom = max(solution)
                    if max_denom < best_max_denom:
                        best_max_denom = max_denom
                        best_solution = solution

    # Phase 4: Fallback greedy if nothing found
    if best_solution is None:
        r = target
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        best_solution = values

    return best_solution
# EVOLVE-BLOCK-END
