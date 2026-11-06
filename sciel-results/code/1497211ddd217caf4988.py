"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    best_max_den = 10000001
    best_result = None
    
    def try_solution(denoms):
        nonlocal best_max_den, best_result
        if not denoms:
            return
        # Check distinct
        if len(set(denoms)) != len(denoms):
            return
        # Check all in range
        for d in denoms:
            if d < 1 or d > 10000000:
                return
        # Check sum
        s = sum(Fraction(1, d) for d in denoms)
        if s == target:
            mx = max(denoms)
            if mx < best_max_den:
                best_max_den = mx
                best_result = list(denoms)
    
    # Check 2-term: 1/a + 1/b = 5/31
    # 1/b = 5/31 - 1/a = (5a - 31)/(31a), so b = 31a/(5a-31)
    # Need 5a - 31 > 0, so a >= 7
    # b = 31a / (5a - 31) must be integer and b > a
    for a in range(7, 10000):
        if time.time() > deadline:
            break
        num = 31 * a
        den = 5 * a - 31
        if den <= 0:
            continue
        if num % den == 0:
            b = num // den
            if b > a:
                try_solution([a, b])
    
    # Check 3-term: 1/a + 1/b + 1/c = 5/31, a < b < c
    # 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a)
    # Let R = (5a - 31)/(31a). Need 1/b + 1/c = R
    # 1/c = R - 1/b = (Rb - 1)/b, so c = b/(Rb - 1)
    # c = b / ((5a-31)*b/(31a) - 1) = b * 31a / ((5a-31)*b - 31a)
    # So (5a-31)*b - 31a must divide 31a*b
    # Alternatively: b*c*(R) = b + c => R*b*c - b - c = 0
    # => (R*b - 1)(R*c - 1) = 1 + R  [standard trick]
    # Or just: for each a, enumerate b in valid range, compute c exactly.
    
    # For a: 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    # Also need 3 terms: 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18
    # But let's be more generous: a from 7 to maybe 100
    for a in range(7, 200):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        # 1/b + 1/c = rem1, b < c
        # 1/b < rem1 => b > 1/rem1
        # 2/b > rem1 => b < 2/rem1
        # So b in (1/rem1, 2/rem1)
        b_min = (rem1.denominator // rem1.numerator) + 1  # floor(1/rem1) + 1
        b_max = (2 * rem1.denominator) // rem1.numerator  # floor(2/rem1)
        if b_min < a + 1:
            b_min = a + 1
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # c = 1/rem2
            if rem2.denominator % rem2.numerator == 0:
                c = rem2.denominator // rem2.numerator
                if c > b:
                    try_solution([a, b, c])
    
    # Check 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
    # This is more expensive. Let's do a limited search.
    # a >= 7, and 4/a > 5/31 => a < 124/5 = 24.8 => a <= 24
    for a in range(7, 25):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        # 1/b + 1/c + 1/d = rem1
        # b > 1/rem1, and 3/b > rem1 => b < 3/rem1
        b_min = (rem1.denominator // rem1.numerator) + 1
        b_max = (3 * rem1.denominator) // rem1.numerator
        if b_min < a + 1:
            b_min = a + 1
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # 1/c + 1/d = rem2
            # c > 1/rem2, 2/c > rem2 => c < 2/rem2
            c_min = (rem2.denominator // rem2.numerator) + 1
            c_max = (2 * rem2.denominator) // rem2.numerator
            if c_min < b + 1:
                c_min = b + 1
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                if rem3.denominator % rem3.numerator == 0:
                    d = rem3.denominator // rem3.numerator
                    if d > c:
                        try_solution([a, b, c, d])
    
    # Check 5-term (limited)
    for a in range(7, 15):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        b_min = (rem1.denominator // rem1.numerator) + 1
        b_max = (4 * rem1.denominator) // rem1.numerator
        if b_min < a + 1:
            b_min = a + 1
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            c_min = (rem2.denominator // rem2.numerator) + 1
            c_max = (3 * rem2.denominator) // rem2.numerator
            if c_min < b + 1:
                c_min = b + 1
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem3 = rem2 - Fraction(1, c)
                if rem3 <= 0:
                    continue
                d_min = (rem3.denominator // rem3.numerator) + 1
                d_max = (2 * rem3.denominator) // rem3.numerator
                if d_min < c + 1:
                    d_min = c + 1
                
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    rem4 = rem3 - Fraction(1, d)
                    if rem4 <= 0:
                        continue
                    if rem4.denominator % rem4.numerator == 0:
                        e = rem4.denominator // rem4.numerator
                        if e > d:
                            try_solution([a, b, c, d, e])
    
    if best_result is None:
        # Fallback to greedy
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_result
# EVOLVE-BLOCK-END
