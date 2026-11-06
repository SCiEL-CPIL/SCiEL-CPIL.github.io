import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    
    # Known solution from seed
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s kill

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == target

    # 1. Exhaustive 3-term search
    # a in [7, 18]
    for a in range(7, 19):
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        # b bounds: 1/b < rem1 => b > 1/rem1
        # 2/b > rem1 => b < 2/rem1
        b_low = int(1 / rem1) + 1
        b_high = int(2 / rem1)
        b_low = max(b_low, a + 1)
        
        if b_low > b_high:
            continue
            
        for b in range(b_low, b_high + 1):
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # c = 1 / rem2
            if rem2.denominator == 1:
                c = rem2.numerator
            else:
                if rem2.numerator % rem2.denominator != 0:
                    continue
                c = rem2.numerator // rem2.denominator
                
            if c <= b:
                continue
            if c < best_max_denom:
                best_solution = [a, b, c]
                best_max_denom = c

    # 2. Search for 4-term solutions with max denominator < best_max_denom
    # 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d < best_max_denom
    # a in [7, 24] (since 4/a > 5/31)
    for a in range(7, 25):
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        # b bounds
        # 1/b < rem1 => b > 1/rem1
        # 3/b > rem1 => b < 3/rem1
        b_low = int(1 / rem1) + 1
        b_high = int(3 / rem1)
        b_low = max(b_low, a + 1)
        
        if b_low > b_high:
            continue
            
        for b in range(b_low, b_high + 1):
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # c bounds
            # 1/c < rem2 => c > 1/rem2
            # 2/c > rem2 => c < 2/rem2
            c_low = int(1 / rem2) + 1
            c_high = int(2 / rem2)
            c_low = max(c_low, b + 1)
            
            if c_low > c_high:
                continue
            
            # We also require d < best_max_denom.
            # d = 1 / (rem2 - 1/c)
            # rem2 - 1/c > 1 / best_max_denom
            # 1/c < rem2 - 1/best_max_denom
            # c > 1 / (rem2 - 1/best_max_denom)
            # This gives a tighter lower bound for c.
            rem_d_limit = rem2 - Fraction(1, best_max_denom)
            if rem_d_limit > 0:
                c_low_tight = int(1 / rem_d_limit) + 1
                c_low = max(c_low, c_low_tight)
            
            if c_low > c_high:
                continue
                
            for c in range(c_low, c_high + 1):
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                # d = 1 / rem3
                if rem3.denominator == 1:
                    d = rem3.numerator
                else:
                    if rem3.numerator % rem3.denominator != 0:
                        continue
                    d = rem3.numerator // rem3.denominator
                    
                if d <= c:
                    continue
                if d < best_max_denom:
                    best_solution = [a, b, c, d]
                    best_max_denom = d

    # 3. Search for 5-term solutions?
    # Likely not needed if 4-term search is efficient and finds something.
    # If 4-term search finds nothing, best_max_denom is still from 3-term.
    # 5-term search space is larger.
    # a in [7, 30].
    # We can add it if time permits.
    
    if time.time() - start_time < time_limit - 5.0:
        # Try 5-term search with max < best_max_denom
        for a in range(7, 31):
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            b_low = int(1 / rem1) + 1
            b_high = int(4 / rem1)
            b_low = max(b_low, a + 1)
            if b_low > b_high:
                continue
            for b in range(b_low, b_high + 1):
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                c_low = int(1 / rem2) + 1
                c_high = int(2 / rem2)
                c_low = max(c_low, b + 1)
                if c_low > c_high:
                    continue
                # Tighten c_low based on d, e < best_max_denom
                # rem2 - 1/c = 1/d + 1/e > 2/best_max_denom
                # 1/c < rem2 - 2/best_max_denom
                rem_de_limit = rem2 - Fraction(2, best_max_denom)
                if rem_de_limit > 0:
                    c_low_tight = int(1 / rem_de_limit) + 1
                    c_low = max(c_low, c_low_tight)
                
                if c_low > c_high:
                    continue
                for c in range(c_low, c_high + 1):
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    # d bounds
                    # 1/d < rem3 => d > 1/rem3
                    # 2/d > rem3 => d < 2/rem3
                    d_low = int(1 / rem3) + 1
                    d_high = int(2 / rem3)
                    d_low = max(d_low, c + 1)
                    if d_low > d_high:
                        continue
                    # e < best_max_denom
                    # rem3 - 1/d > 1/best_max_denom
                    # 1/d < rem3 - 1/best_max_denom
                    # d > 1 / (rem3 - 1/best_max_denom)
                    rem_e_limit = rem3 - Fraction(1, best_max_denom)
                    if rem_e_limit > 0:
                        d_low_tight = int(1 / rem_e_limit) + 1
                        d_low = max(d_low, d_low_tight)
                    
                    if d_low > d_high:
                        continue
                    for d in range(d_low, d_high + 1):
                        rem4 = rem3 - Fraction(1, d)
                        if rem4 <= 0:
                            continue
                        if rem4.denominator == 1:
                            e = rem4.numerator
                        else:
                            if rem4.numerator % rem4.denominator != 0:
                                continue
                            e = rem4.numerator // rem4.denominator
                        if e <= d:
                            continue
                        if e < best_max_denom:
                            best_solution = [a, b, c, d, e]
                            best_max_denom = e

    if verify(best_solution):
        return best_solution
        
    # Fallback
    return [8, 28, 1736]
