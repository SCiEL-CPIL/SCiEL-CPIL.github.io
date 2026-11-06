# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    def check_and_update(dens):
        nonlocal best, best_max
        mx = max(dens)
        if mx < best_max:
            best_max = mx
            best = list(dens)
    
    # Meet-in-the-middle: precompute sums of 2 and 3 unit fractions
    # with denominators up to a reasonable bound
    MITM_BOUND = 100  # denominators up to 100 for precomputation
    
    # Precompute all pairs (a, b) with a < b <= MITM_BOUND
    # Store: (numerator, denominator) -> list of denominator sets
    pair_sums = defaultdict(list)  # (p, q) -> [(a, b), ...]
    for a in range(2, MITM_BOUND + 1):
        for b in range(a + 1, MITM_BOUND + 1):
            p = a + b
            q = a * b
            g = gcd(p, q)
            p //= g
            q //= g
            pair_sums[(p, q)].append((a, b))
    
    # Precompute all triples (a, b, c) with a < b < c <= MITM_BOUND
    triple_sums = defaultdict(list)  # (p, q) -> [(a, b, c), ...]
    for a in range(2, MITM_BOUND + 1):
        for b in range(a + 1, MITM_BOUND + 1):
            for c in range(b + 1, MITM_BOUND + 1):
                # 1/a + 1/b + 1/c = (bc + ac + ab) / (abc)
                num = b * c + a * c + a * b
                den = a * b * c
                g = gcd(num, den)
                num //= g
                den //= g
                triple_sums[(num, den)].append((a, b, c))
    
    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # For each a, compute remainder 5/31 - 1/a, then look for pair in pair_sums
    for a in range(7, min(50, best_max)):
        if time.time() > deadline:
            break
        # Remainder: 5/31 - 1/a = (5a - 31) / (31a)
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        
        # Look for pair (b, c) with b > a, c > b, and 1/b + 1/c = p_rem/q_rem
        # Also need b, c < best_max
        # Check if this remainder is in pair_sums
        if (p_rem, q_rem) in pair_sums:
            for (b, c) in pair_sums[(p_rem, q_rem)]:
                if b > a and c > b and c < best_max:
                    check_and_update([a, b, c])
        
        # Also do direct search for 3-term (in case b or c > MITM_BOUND)
        b_lo = a + 1
        b_hi = min(best_max - 1, (2 * q_rem) // p_rem)
        for b in range(b_lo, b_hi + 1):
            if time.time() > deadline:
                break
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            if q2 % p2 != 0:
                continue
            c = q2 // p2
            if c > b and c < best_max:
                check_and_update([a, b, c])
    
    # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # For each a, compute remainder, then look for triple in triple_sums
    for a in range(7, min(50, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        
        # Look for triple (b, c, d) with b > a, c > b, d > c, and sum = p_rem/q_rem
        if (p_rem, q_rem) in triple_sums:
            for (b, c, d) in triple_sums[(p_rem, q_rem)]:
                if b > a and c > b and d > c and d < best_max:
                    check_and_update([a, b, c, d])
        
        # Also do direct 4-term search with limited range
        b_lo = a + 1
        b_hi = min(best_max - 1, (3 * q_rem) // p_rem)
        for b in range(b_lo, min(b_hi + 1, a + 20)):  # Limit inner loops
            if time.time() > deadline:
                break
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            g2 = gcd(p2, q2)
            p2 //= g2
            q2 //= g2
            
            c_lo = b + 1
            c_hi = min(best_max - 1, (2 * q2) // p2)
            for c in range(c_lo, min(c_hi + 1, b + 15)):
                if time.time() > deadline:
                    break
                p3 = p2 * c - q2
                if p3 <= 0:
                    continue
                q3 = q2 * c
                if q3 % p3 != 0:
                    continue
                d = q3 // p3
                if d > c and d < best_max:
                    check_and_update([a, b, c, d])
    
    # Try 5-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e = 5/31
    # For each a, compute remainder, then look for quadruple... 
    # Instead, use pair+triple meet-in-middle: 1/a + (pair) + (triple) or similar
    # Simpler: for each a, b, compute remainder, look for triple in triple_sums
    for a in range(7, min(40, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        
        b_lo = a + 1
        b_hi = min(best_max - 1, (4 * q_rem) // p_rem)
        for b in range(b_lo, min(b_hi + 1, a + 15)):
            if time.time() > deadline:
                break
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            g2 = gcd(p2, q2)
            p2 //= g2
            q2 //= g2
            
            # Look for triple (c, d, e) with c > b, sum = p2/q2
            if (p2, q2) in triple_sums:
                for (c, d, e) in triple_sums[(p2, q2)]:
                    if c > b and d > c and e > d and e < best_max:
                        check_and_update([a, b, c, d, e])
            
            # Also direct search for 5-term with tight bounds
            c_lo = b + 1
            c_hi = min(best_max - 1, (3 * q2) // p2)
            for c in range(c_lo, min(c_hi + 1, b + 10)):
                if time.time() > deadline:
                    break
                p3 = p2 * c - q2
                if p3 <= 0:
                    continue
                q3 = q2 * c
                g3 = gcd(p3, q3)
                p3 //= g3
                q3 //= g3
                
                d_lo = c + 1
                d_hi = min(best_max - 1, (2 * q3) // p3)
                for d in range(d_lo, min(d_hi + 1, c + 8)):
                    if time.time() > deadline:
                        break
                    p4 = p3 * d - q3
                    if p4 <= 0:
                        continue
                    q4 = q3 * d
                    if q4 % p4 != 0:
                        continue
                    e = q4 // p4
                    if e > d and e < best_max:
                        check_and_update([a, b, c, d, e])
    
    return best
# EVOLVE-BLOCK-END
