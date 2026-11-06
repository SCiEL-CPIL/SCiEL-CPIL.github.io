"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    deadline = time.time() + 55
    target = Fraction(5, 31)
    
    best_max = 248  # known solution: 8, 31, 248
    best_solution = [8, 31, 248]
    
    # For 1/a + 1/b = p/q (lowest terms): (pa-q)(pb-q) = q^2
    # Enumerate divisors u of q^2 with u <= q, check divisibility conditions.
    
    def find_two_term(p, q, min_a, max_b):
        """Find (a,b) with min_a <= a <= b <= max_b, 1/a+1/b = p/q (reduced)."""
        q2 = q * q
        u_min = max(1, p * min_a - q)
        v_max = p * max_b - q
        if v_max <= 0:
            return None
        u_lo = max(u_min, (q2 + v_max - 1) // v_max)
        u_hi = min(q, v_max)
        for u in range(u_lo, u_hi + 1):
            if q2 % u != 0:
                continue
            v = q2 // u
            if (u + q) % p != 0 or (v + q) % p != 0:
                continue
            a = (u + q) // p
            b = (v + q) // p
            if min_a <= a <= b <= max_b:
                return (a, b)
        return None
    
    # n=3: 1/a+1/b+1/c = 5/31, a<=b<=c
    for c in range(7, best_max):
        if time.time() > deadline:
            break
        p = 5 * c - 31
        q = 31 * c
        if p <= 0:
            continue
        g = gcd(p, q)
        p_r, q_r = p // g, q // g
        pair = find_two_term(p_r, q_r, 7, c)
        if pair:
            a, b = pair
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                best_max = c
                best_solution = [a, b, c]
                break
    
    # n=4: 1/a+1/b+1/c+1/d = 5/31, a<=b<=c<=d
    for d in range(7, best_max):
        if time.time() > deadline:
            break
        P = 5 * d - 31
        Q = 31 * d
        if P <= 0:
            continue
        g = gcd(P, Q)
        P_r, Q_r = P // g, Q // g
        
        found = False
        c_min = max(7, (Q_r + P_r - 1) // P_r)
        for c in range(c_min, d + 1):
            if time.time() > deadline:
                break
            p2 = P_r * c - Q_r
            if p2 <= 0:
                continue
            q2 = Q_r * c
            g2 = gcd(p2, q2)
            p2_r, q2_r = p2 // g2, q2 // g2
            pair = find_two_term(p2_r, q2_r, 7, c)
            if pair:
                a, b = pair
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                    best_max = d
                    best_solution = [a, b, c, d]
                    found = True
                    break
        if found:
            break
    
    # Final verification
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            best_solution = [8, 31, 248]
            best_max = 248
    
    return best_solution
# EVOLVE-BLOCK-END
