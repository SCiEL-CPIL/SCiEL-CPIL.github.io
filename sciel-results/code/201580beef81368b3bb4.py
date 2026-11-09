"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from collections import defaultdict

def construct():
    """
    Returns the set of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    Uses divisor-based enumeration for 3 terms and meet-in-the-middle for 4 terms.
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    best_max_den = 248
    best_solution = [8, 31, 248]

    def find_3_terms(max_c):
        """Find 1/a + 1/b + 1/c = 5/31 with a<b<c<=max_c using divisor structure."""
        nonlocal best_max_den, best_solution
        if time.time() > deadline:
            return False
        # 1/a + 1/b + 1/c = 5/31
        # => 31ab + 31ac + 31bc = 5abc
        # => 5abc - 31ab - 31ac - 31bc = 0
        # Multiply by 5: 25abc - 155ab - 155ac - 155bc = 0
        # Add 31^2 = 961 to complete: (5a-31)(5b-31)(5c-31) related form
        # Better: fix a, then 1/b + 1/c = (5/31 - 1/a) = (5a-31)/(31a)
        # => 31a(b+c) = (5a-31)bc
        # => (5a-31)bc - 31ab - 31ac = 0
        # => b(5a-31)c - 31a(b+c) = 0
        # => b(5a-31)c - 31ab - 31ac = 0
        # For fixed a: 1/b + 1/c = (5a-31)/(31a)
        # => c = 31ab / ((5a-31)b - 31a)
        # Let D = (5a-31)b - 31a. Need D > 0 and D | 31ab.
        # D = (5a-31)b - 31a
        # 31ab = D * c => c = 31ab/D
        # For c to be integer: D | 31ab
        # Also c > b means 31ab/D > b => 31a > D => 31a > (5a-31)b - 31a => 62a > (5a-31)b => b < 62a/(5a-31)
        # And D > 0 means (5a-31)b > 31a => b > 31a/(5a-31)
        
        for a in range(7, min(max_c, 1000) + 1):
            if time.time() > deadline:
                return False
            five_a_31 = 5 * a - 31
            if five_a_31 <= 0:
                continue
            # b > 31a/(5a-31) and b < 62a/(5a-31) and b > a
            b_lo = max(a + 1, (31 * a) // five_a_31 + 1)
            b_hi = min(max_c - 1, (62 * a) // five_a_31)  # c > b requires b < 62a/(5a-31)
            if b_lo > b_hi:
                continue
            for b in range(b_lo, b_hi + 1):
                if time.time() > deadline:
                    return False
                D = five_a_31 * b - 31 * a
                if D <= 0:
                    continue
                num = 31 * a * b
                if num % D != 0:
                    continue
                c = num // D
                if c <= b or c > max_c:
                    continue
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == Fraction(5, 31):
                    sol = [a, b, c]
                    if max(sol) < best_max_den:
                        best_max_den = max(sol)
                        best_solution = sol
                    return True
        return False

    # Search 3-term solutions
    for mc in range(best_max_den - 1, 6, -1):
        if time.time() > deadline:
            break
        if find_3_terms(mc):
            if best_max_den <= mc:
                break

    # 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31
    # Meet-in-the-middle: precompute all pairs (a,b) with a<b, store sum 1/a+1/b
    # Then for each pair (c,d) with c>d, check if 1/a+1/b = 5/31 - 1/c - 1/d
    # Actually: for pairs (a,b) with a<b, and pairs (c,d) with b<c<d, check sum = 5/31
    
    def find_4_terms(max_d):
        nonlocal best_max_den, best_solution
        if time.time() > deadline:
            return False
        # Precompute all valid (a,b) pairs with a < b <= max_d
        # Store by their Fraction sum
        pair_sums = defaultdict(list)
        for a in range(7, max_d + 1):
            if time.time() > deadline:
                return False
            if 5 * a - 31 <= 0:
                continue
            b_min = max(a + 1, (31 * a) // (5 * a - 31) + 1)
            # Need 1/a + 1/b < 5/31 (since we still need c,d positive)
            # 1/b < 5/31 - 1/a => b > 31a/(5a-31)
            for b in range(b_min, max_d + 1):
                if time.time() > deadline:
                    return False
                s = Fraction(1, a) + Fraction(1, b)
                if s >= Fraction(5, 31):
                    break
                pair_sums[s].append((a, b))
        
        # Now for each (c,d) pair with c < d <= max_d, check if 5/31 - 1/c - 1/d is in pair_sums
        # And the (a,b) pair must satisfy a < b < c
        found = False
        for c in range(7, max_d + 1):
            if time.time() > deadline:
                return False
            if 5 * c - 31 <= 0:
                continue
            d_min = max(c + 1, (31 * c) // (5 * c - 31) + 1)
            for d in range(d_min, max_d + 1):
                if time.time() > deadline:
                    return False
                rem = Fraction(5, 31) - Fraction(1, c) - Fraction(1, d)
                if rem <= 0:
                    break
                if rem in pair_sums:
                    for (a, b) in pair_sums[rem]:
                        if b < c:  # Ensure a < b < c < d
                            sol = [a, b, c, d]
                            if max(sol) < best_max_den:
                                best_max_den = max(sol)
                                best_solution = sol
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        return found

    for md in range(best_max_den - 1, 6, -1):
        if time.time() > deadline:
            break
        if find_4_terms(md):
            if best_max_den <= md:
                break

    # 5 terms: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
    # Split: (a,b) + (c,d,e) = 5/31 or (a,b,c) + (d,e) = 5/31
    # Use (a,b) pairs and (d,e) pairs, with middle term c
    # Actually: for fixed (a,b), need 1/c + 1/d + 1/e = 5/31 - 1/a - 1/b
    # This is a 3-term problem for the remainder
    
    def find_5_terms(max_e):
        nonlocal best_max_den, best_solution
        if time.time() > deadline:
            return False
        # For each (a,b) pair, solve 3-term for remainder
        for a in range(7, max_e + 1):
            if time.time() > deadline:
                return False
            if 5 * a - 31 <= 0:
                continue
            b_min = max(a + 1, (31 * a) // (5 * a - 31) + 1)
            for b in range(b_min, max_e + 1):
                if time.time() > deadline:
                    return False
                rem = Fraction(5, 31) - Fraction(1, a) - Fraction(1, b)
                if rem <= 0:
                    break
                # Solve 1/c + 1/d + 1/e = rem with b < c < d < e <= max_e
                # c > 1/rem... actually c > 1/(rem/3) roughly, but precisely:
                # 1/c < rem => c > 1/rem
                # Also 1/c + 1/d + 1/e = rem, c < d < e
                # Max of 1/c+1/d+1/e for c>=c_min is when c is smallest
                # Need 3/c_max_possible >= rem, i.e., c <= 3/rem
                c_lo = max(b + 1, 1 // rem + 1) if rem < 1 else b + 1
                c_hi = min(max_e - 2, 3 // rem) if rem > 0 else max_e - 2
                if c_lo > c_hi:
                    continue
                for c in range(c_lo, c_hi + 1):
                    if time.time() > deadline:
                        return False
                    rem2 = rem - Fraction(1, c)
                    if rem2 <= 0:
                        continue
                    # 1/d + 1/e = rem2, d > c, d < e
                    # e = d / (rem2*d - 1)... 
                    # 1/d + 1/e = rem2 => e = d/(rem2*d - 1)
                    # rem2*d - 1 > 0 => d > 1/rem2
                    # d > c, and e > d => d/(rem2*d-1) > d => 1 > rem2*d - 1 => rem2*d < 2 => d < 2/rem2
                    d_lo = max(c + 1, 1 // rem2 + 1) if rem2 < 1 else c + 1
                    d_hi = min(max_e - 1, 2 // rem2) if rem2 > 0 else max_e - 1
                    if d_lo > d_hi:
                        continue
                    for d in range(d_lo, d_hi + 1):
                        if time.time() > deadline:
                            return False
                        # e = d / (rem2*d - 1)
                        # rem2 = p/q in lowest terms
                        # e = d*q / (p*d - q)
                        p = rem2.numerator
                        q = rem2.denominator
                        denom_e = p * d - q
                        if denom_e <= 0:
                            continue
                        if (d * q) % denom_e != 0:
                            continue
                        e = (d * q) // denom_e
                        if e <= d or e > max_e:
                            continue
                        # Verify
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == Fraction(5, 31):
                            sol = [a, b, c, d, e]
                            if max(sol) < best_max_den:
                                best_max_den = max(sol)
                                best_solution = sol
                            return True
        return False

    for me in range(best_max_den - 1, 6, -1):
        if time.time() > deadline:
            break
        if find_5_terms(me):
            if best_max_den <= me:
                break

    return best_solution
# EVOLVE-BLOCK-END
