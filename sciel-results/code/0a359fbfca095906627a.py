import time
from fractions import Fraction
from math import isqrt

def construct():
    start_time = time.time()
    target = Fraction(5, 31)
    deadline = start_time + 80.0  # Safety margin before 90s kill

    def find_two_terms_fast(rem, min_d, max_d):
        """Find distinct b, c in [min_d, max_d] with 1/b + 1/c = rem using divisor method."""
        if rem <= 0:
            return None
        p = rem.numerator
        q = rem.denominator
        q2 = q * q
        best = None
        for i in range(1, isqrt(q2) + 1):
            if q2 % i == 0:
                for d in (i, q2 // i):
                    if (d + q) % p != 0:
                        continue
                    b = (d + q) // p
                    d2 = q2 // d
                    if (d2 + q) % p != 0:
                        continue
                    c = (d2 + q) // p
                    if b <= 0 or c <= 0:
                        continue
                    if b > c:
                        b, c = c, b
                    if b == c:
                        continue
                    if b < min_d or c > max_d:
                        continue
                    if best is None or c < best[1]:
                        best = (b, c)
        return best

    def check_three_terms(M):
        """Check if there exist a < b < c <= M with 1/a+1/b+1/c = 5/31."""
        # a must be at least ceil(31/5) = 7 (since 1/a < 5/31 means a > 31/5)
        # and at most floor(3*31/5) = 18 (since 3/a >= 5/31 means a <= 93/5)
        a_min = max(1, (31 + 4) // 5 + 1)  # a > 31/5, so a >= 7
        a_max = min(M, 93 // 5)  # a <= 93/5 = 18.6, so a <= 18
        for a in range(a_min, a_max + 1):
            rem2 = target - Fraction(1, a)
            if rem2 <= 0:
                break
            # b > a, and 1/b < rem2 means b > 1/rem2
            # Also 2/b >= rem2 means b <= 2/rem2
            b_min = a + 1
            b_max = min(M, (2 * rem2.denominator) // rem2.numerator)
            if b_max < b_min:
                continue
            # Use divisor method for the pair (b, c)
            pair = find_two_terms_fast(rem2, b_min, M)
            if pair is not None:
                b, c = pair
                if b > a and c > b and c <= M:
                    return [a, b, c]
        return None

    def check_four_terms(M):
        """Check if there exist a < b < c < d <= M with sum = 5/31."""
        a_min = max(1, (31 + 3) // 4 + 1)  # a > 31/4 = 7.75, so a >= 8
        a_max = min(M, 124 // 5)  # a <= 124/5 = 24.8, so a <= 24
        for a in range(a_min, a_max + 1):
            rem2 = target - Fraction(1, a)
            if rem2 <= 0:
                break
            b_min = a + 1
            b_max = min(M, (3 * rem2.denominator) // rem2.numerator)
            if b_max < b_min:
                continue
            for b in range(b_min, b_max + 1):
                rem3 = rem2 - Fraction(1, b)
                if rem3 <= 0:
                    break
                c_min = b + 1
                c_max = min(M, (2 * rem3.denominator) // rem3.numerator)
                if c_max < c_min:
                    continue
                pair = find_two_terms_fast(rem3, c_min, M)
                if pair is not None:
                    c, d = pair
                    if c > b and d > c and d <= M:
                        return [a, b, c, d]
        return None

    def check_five_terms(M):
        """Check if there exist a < b < c < d < e <= M with sum = 5/31."""
        a_min = max(1, (31 + 4) // 5 + 1)  # a >= 7
        a_max = min(M, 155 // 5)  # a <= 31
        for a in range(a_min, a_max + 1):
            if time.time() > deadline:
                return None
            rem2 = target - Fraction(1, a)
            if rem2 <= 0:
                break
            b_min = a + 1
            b_max = min(M, (4 * rem2.denominator) // rem2.numerator)
            if b_max < b_min:
                continue
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return None
                rem3 = rem2 - Fraction(1, b)
                if rem3 <= 0:
                    break
                c_min = b + 1
                c_max = min(M, (3 * rem3.denominator) // rem3.numerator)
                if c_max < c_min:
                    continue
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return None
                    rem4 = rem3 - Fraction(1, c)
                    if rem4 <= 0:
                        break
                    d_min = c + 1
                    d_max = min(M, (2 * rem4.denominator) // rem4.numerator)
                    if d_max < d_min:
                        continue
                    pair = find_two_terms_fast(rem4, d_min, M)
                    if pair is not None:
                        d, e = pair
                        if d > c and e > d and e <= M:
                            return [a, b, c, d, e]
        return None

    # Try increasing M starting from small values
    # First check 2-term (we know it doesn't exist, but verify)
    # Then 3-term, 4-term, 5-term
    
    for M in range(7, 250):
        if time.time() > deadline:
            break
        
        # Check 3-term
        sol = check_three_terms(M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return sol
        
        # Check 4-term
        sol = check_four_terms(M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return sol
        
        # Check 5-term
        sol = check_five_terms(M)
        if sol is not None:
            if sum(Fraction(1, x) for x in sol) == target:
                return sol
    
    # Fallback
    return [8, 31, 248]
