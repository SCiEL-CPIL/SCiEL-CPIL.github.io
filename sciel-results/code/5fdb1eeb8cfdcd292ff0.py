"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Known solution
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Try 2-term solutions: 1/a + 1/b = 5/31
    # 31(a+b) = 5ab => 5ab - 31a - 31b = 0 => (5a-31)(5b-31) = 961
    # 961 = 31^2, factors: 1,31,961
    # 5a-31 = 1 => a = 32/5 not integer
    # 5a-31 = 31 => a = 62/5 not integer  
    # 5a-31 = 961 => a = 992/5 not integer
    # No 2-term solution with integer denominators
    
    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31 with a < b < c
    # a must satisfy: 1/a < 5/31 => a > 6.2 => a >= 7
    # a must satisfy: 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18
    # But also need a < best_max for improvement
    for a in range(7, min(19, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # Need 1/b + 1/c = rem_a with b > a, c > b
        # b must satisfy: 2/b > rem_a => b < 2/rem_a
        # b must satisfy: 1/b < rem_a => b > 1/rem_a
        # Also b > a
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
        b_max = min(best_max - 1, int(2 * rem_a.denominator / rem_a.numerator))
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            # c = 1/rem_b
            if rem_b.numerator == 1:
                c = rem_b.denominator
                if c > b and c < best_max:
                    best_max = c
                    best_solution = [a, b, c]
    
    # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a: 4/a > 5/31 => a < 124/5 = 24.8 => a <= 24
    # a: 1/a < 5/31 => a >= 7
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # b: 3/b > rem_a => b < 3/rem_a
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
        b_max = min(best_max - 1, int(3 * rem_a.denominator / rem_a.numerator))
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            # c: 2/c > rem_b => c < 2/rem_b
            c_min = max(b + 1, (rem_b.denominator + rem_b.numerator - 1) // rem_b.numerator + 1)
            c_max = min(best_max - 1, int(2 * rem_b.denominator / rem_b.numerator))
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                # d = 1/rem_c
                if rem_c.numerator == 1:
                    d = rem_c.denominator
                    if d > c and d < best_max:
                        best_max = d
                        best_solution = [a, b, c, d]
    
    # Try 5-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
    # a: 5/a > 5/31 => a < 31 => a <= 30
    # a: 1/a < 5/31 => a >= 7
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # b: 4/b > rem_a => b < 4/rem_a
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
        b_max = min(best_max - 1, int(4 * rem_a.denominator / rem_a.numerator))
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            # c: 3/c > rem_b => c < 3/rem_b
            c_min = max(b + 1, (rem_b.denominator + rem_b.numerator - 1) // rem_b.numerator + 1)
            c_max = min(best_max - 1, int(3 * rem_b.denominator / rem_b.numerator))
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                # d: 2/d > rem_c => d < 2/rem_c
                d_min = max(c + 1, (rem_c.denominator + rem_c.numerator - 1) // rem_c.numerator + 1)
                d_max = min(best_max - 1, int(2 * rem_c.denominator / rem_c.numerator))
                
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        break
                    # e = 1/rem_d
                    if rem_d.numerator == 1:
                        e = rem_d.denominator
                        if e > d and e < best_max:
                            best_max = e
                            best_solution = [a, b, c, d, e]
    
    # Try 6-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e + 1/f = 5/31
    # a: 6/a > 5/31 => a < 186/5 = 37.2 => a <= 37
    # a: 1/a < 5/31 => a >= 7
    for a in range(7, min(38, best_max)):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # b: 5/b > rem_a => b < 5/rem_a
        b_min = max(a + 1, (rem_a.denominator + rem_a.numerator - 1) // rem_a.numerator + 1)
        b_max = min(best_max - 1, int(5 * rem_a.denominator / rem_a.numerator))
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            # c: 4/c > rem_b => c < 4/rem_b
            c_min = max(b + 1, (rem_b.denominator + rem_b.numerator - 1) // rem_b.numerator + 1)
            c_max = min(best_max - 1, int(4 * rem_b.denominator / rem_b.numerator))
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                # d: 3/d > rem_c => d < 3/rem_c
                d_min = max(c + 1, (rem_c.denominator + rem_c.numerator - 1) // rem_c.numerator + 1)
                d_max = min(best_max - 1, int(3 * rem_c.denominator / rem_c.numerator))
                
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    rem_d = rem_c - Fraction(1, d)
                    if rem_d <= 0:
                        break
                    # e: 2/e > rem_d => e < 2/rem_d
                    e_min = max(d + 1, (rem_d.denominator + rem_d.numerator - 1) // rem_d.numerator + 1)
                    e_max = min(best_max - 1, int(2 * rem_d.denominator / rem_d.numerator))
                    
                    for e in range(e_min, e_max + 1):
                        if time.time() > deadline:
                            break
                        rem_e = rem_d - Fraction(1, e)
                        if rem_e <= 0:
                            break
                        # f = 1/rem_e
                        if rem_e.numerator == 1:
                            f = rem_e.denominator
                            if f > e and f < best_max:
                                best_max = f
                                best_solution = [a, b, c, d, e, f]
    
    return best_solution
# EVOLVE-BLOCK-END
