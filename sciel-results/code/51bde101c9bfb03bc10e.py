"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    
    # Known solution: 5/31 = 1/8 + 1/31 + 1/248, max = 248
    best = [8, 31, 248]
    best_max = 248
    
    # Verify
    assert (31*248 + 8*248 + 8*31) * 31 == 5 * 8 * 31 * 248, "Known solution invalid"
    
    target_num, target_den = 5, 31
    
    def check_and_update(dens):
        nonlocal best, best_max
        mx = max(dens)
        if mx < best_max:
            best_max = mx
            best = list(dens)
    
    def frac_sub(p1, q1, p2, q2):
        """Return (p, q) for p1/q1 - p2/q2 in reduced form."""
        p = p1 * q2 - p2 * q1
        q = q1 * q2
        if p == 0:
            return (0, 1)
        g = gcd(abs(p), q)
        return (p // g, q // g)
    
    def frac_add_inv(p, q, a):
        """Return (p, q) for p/q - 1/a in reduced form."""
        return frac_sub(p, q, 1, a)
    
    # Try 2-term solutions: 1/a + 1/b = 5/31 with a < b
    # 1/a + 1/b = 5/31 → 31(a+b) = 5ab → 5ab - 31a - 31b = 0
    # (5a - 31)(5b - 31) = 31^2 = 961
    # So we need factors of 961: 1, 31, 961
    for fa in [1, 31, 961]:
        if 961 % fa == 0:
            fb = 961 // fa
            a = (fa + 31) // 5
            b = (fb + 31) // 5
            if (fa + 31) % 5 == 0 and (fb + 31) % 5 == 0:
                if a > 0 and b > 0 and a < b and b < best_max:
                    check_and_update([a, b])
    
    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31 with a < b < c
    # a must satisfy 1/a < 5/31 → a > 31/5 = 6.2, so a >= 7
    # Also 3/a > 5/31 → a < 93/5 = 18.6, so a <= 18 for 3 terms
    for a in range(7, min(19, best_max)):
        if time.time() > deadline:
            break
        # rem = 5/31 - 1/a = (5a - 31)/(31a)
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        
        # Need 1/b + 1/c = p_rem/q_rem with b > a, c > b
        # b > q_rem/p_rem and b < 2*q_rem/p_rem
        b_lo = a + 1
        b_hi = min(best_max - 1, (2 * q_rem) // p_rem)
        for b in range(b_lo, b_hi):
            if time.time() > deadline:
                break
            # rem2 = p_rem/q_rem - 1/b = (p_rem*b - q_rem)/(q_rem*b)
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            # c = q2/p2 must be integer
            if q2 % p2 != 0:
                continue
            c = q2 // p2
            if c > b and c < best_max:
                check_and_update([a, b, c])
    
    # Try 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a >= 7, and 4/a > 5/31 → a < 124/5 = 24.8, so a <= 24
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            break
        p_rem = 5 * a - 31
        q_rem = 31 * a
        if p_rem <= 0:
            continue
        g = gcd(p_rem, q_rem)
        p_rem //= g
        q_rem //= g
        
        # b > q_rem/p_rem, and 3/b > p_rem/q_rem → b < 3*q_rem/p_rem
        b_lo = a + 1
        b_hi = min(best_max - 1, (3 * q_rem) // p_rem)
        for b in range(b_lo, b_hi):
            if time.time() > deadline:
                break
            p2 = p_rem * b - q_rem
            if p2 <= 0:
                continue
            q2 = q_rem * b
            g2 = gcd(p2, q2)
            p2 //= g2
            q2 //= g2
            
            # c > q2/p2, and 2/c > p2/q2 → c < 2*q2/p2
            c_lo = b + 1
            c_hi = min(best_max - 1, (2 * q2) // p2)
            for c in range(c_lo, c_hi):
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
    # a >= 7, and 5/a > 5/31 → a < 31, so a <= 30
    for a in range(7, min(31, best_max)):
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
        for b in range(b_lo, b_hi):
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
            c_hi = min(best_max - 1, (3 * q2) // p2)
            for c in range(c_lo, c_hi):
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
                for d in range(d_lo, d_hi):
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
    
    # Try 6-term solutions: 1/a + 1/b + 1/c + 1/d + 1/e + 1/f = 5/31
    # a >= 7, and 6/a > 5/31 → a < 186/5 = 37.2, so a <= 37
    for a in range(7, min(38, best_max)):
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
        b_hi = min(best_max - 1, (5 * q_rem) // p_rem)
        for b in range(b_lo, b_hi):
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
            c_hi = min(best_max - 1, (4 * q2) // p2)
            for c in range(c_lo, c_hi):
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
                d_hi = min(best_max - 1, (3 * q3) // p3)
                for d in range(d_lo, d_hi):
                    if time.time() > deadline:
                        break
                    p4 = p3 * d - q3
                    if p4 <= 0:
                        continue
                    q4 = q3 * d
                    g4 = gcd(p4, q4)
                    p4 //= g4
                    q4 //= g4
                    
                    e_lo = d + 1
                    e_hi = min(best_max - 1, (2 * q4) // p4)
                    for e in range(e_lo, e_hi):
                        if time.time() > deadline:
                            break
                        p5 = p4 * e - q4
                        if p5 <= 0:
                            continue
                        q5 = q4 * e
                        if q5 % p5 != 0:
                            continue
                        f = q5 // p5
                        if f > e and f < best_max:
                            check_and_update([a, b, c, d, e, f])
    
    return best
# EVOLVE-BLOCK-END
