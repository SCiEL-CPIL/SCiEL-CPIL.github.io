"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    target = Fraction(5, 31)
    
    best_max = None
    best_result = None
    
    # Try 2-term: 1/a + 1/b = 5/31
    # (5a-31)(5b-31) = 961
    # Check divisors of 961 = 31^2
    for d in [1, 31, 961]:
        a = (31 + d) / 5
        b = (31 + 961 // d) / 5
        if a == int(a) and b == int(b) and a > 0 and b > 0 and int(a) != int(b):
            a, b = int(a), int(b)
            if a < b:
                if 1 <= a <= 10000000 and 1 <= b <= 10000000:
                    result = [a, b]
                    mx = max(result)
                    if best_max is None or mx < best_max:
                        best_max = mx
                        best_result = result
    
    # Try 3-term: 1/a + 1/b + 1/c = 5/31
    # Enumerate a < b, solve for c
    max_a = 200  # small denominators first
    for a in range(1, max_a + 1):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        # Now 1/b + 1/c = r1
        # (r1.den * b - r1.num)(r1.den * c - r1.num) = r1.num^2 ... 
        # Simpler: 1/b + 1/c = r1/n1 where r1 = p/q
        # 1/b + 1/c = p/q → q(b+c) = pbc → pbc - qb - qc = 0
        # (pb - q)(pc - q) = q^2
        p, q = r1.numerator, r1.denominator
        # Enumerate divisors of q^2
        q2 = q * q
        # Find all divisors of q2
        divs = []
        for i in range(1, int(q2**0.5) + 1):
            if q2 % i == 0:
                divs.append(i)
                if i != q2 // i:
                    divs.append(q2 // i)
        for d1 in divs:
            d2 = q2 // d1
            # pb - q = d1 → b = (q + d1) / p
            # pc - q = d2 → c = (q + d2) / p
            if p == 0:
                continue
            b_num = q + d1
            c_num = q + d2
            if b_num % p == 0 and c_num % p == 0:
                b = b_num // p
                c = c_num // p
                if b > 0 and c > 0 and b != c and b != a and c != a:
                    if b < c:
                        if b < a:
                            b, c = c, b
                        if b > a and c > b:
                            if 1 <= a <= 10000000 and 1 <= b <= 10000000 and 1 <= c <= 10000000:
                                result = [a, b, c]
                                mx = max(result)
                                if best_max is None or mx < best_max:
                                    best_max = mx
                                    best_result = result
    
    # Try 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
    # Enumerate a < b < c, solve for d
    # This is more expensive, so limit the range
    if time.time() < deadline - 10:
        max_a4 = 50
        max_b4 = 100
        for a in range(1, max_a4 + 1):
            if time.time() > deadline - 5:
                break
            r1 = target - Fraction(1, a)
            if r1 <= 0:
                continue
            for b in range(a + 1, max_b4 + 1):
                if time.time() > deadline - 5:
                    break
                r2 = r1 - Fraction(1, b)
                if r2 <= 0:
                    continue
                # 1/c + 1/d = r2
                p, q = r2.numerator, r2.denominator
                if p == 0:
                    continue
                q2 = q * q
                divs = []
                for i in range(1, int(q2**0.5) + 1):
                    if q2 % i == 0:
                        divs.append(i)
                        if i != q2 // i:
                            divs.append(q2 // i)
                for d1 in divs:
                    d2 = q2 // d1
                    c_num = q + d1
                    d_num = q + d2
                    if c_num % p == 0 and d_num % p == 0:
                        c = c_num // p
                        d = d_num // p
                        if c > 0 and d > 0 and c != d and c != a and c != b and d != a and d != b:
                            if c < d:
                                if c < b:
                                    c, d = d, c
                            if c > b and d > c:
                                if 1 <= a <= 10000000 and 1 <= b <= 10000000 and 1 <= c <= 10000000 and 1 <= d <= 10000000:
                                    result = [a, b, c, d]
                                    mx = max(result)
                                    if best_max is None or mx < best_max:
                                        best_max = mx
                                        best_result = result
    
    # If we found nothing, fall back to greedy
    if best_result is None:
        r = target
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_result
# EVOLVE-BLOCK-END
