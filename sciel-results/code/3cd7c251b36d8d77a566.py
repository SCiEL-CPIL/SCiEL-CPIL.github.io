"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    target = Fraction(5, 31)
    
    # Known optimal 3-term solution: {7, 62, 434}
    # 1/7 + 1/62 + 1/434 = 35/217 + 5/217 + 1/434 = 40/217 + 1/434
    # = 80/434 + 1/434 = 81/434... wait let me recompute
    # 1/7 + 1/62 = (62+7)/434 = 69/434
    # 69/434 + 1/434 = 70/434 = 35/217
    # 35/217 = 5*7/31*7 = 35/217. And 5/31 = 35/217. Yes!
    # So {7, 62, 434} sums to exactly 5/31. Max denominator = 434.
    
    # Let's verify: 5/31 = 35/217. 1/7 = 31/217. 1/62 = 3.5/217... no.
    # Let me use exact fractions.
    # 1/7 + 1/62 + 1/434
    # LCM of 7, 62, 434: 62 = 2*31, 434 = 2*7*31, 7=7
    # LCM = 2*7*31 = 434
    # 1/7 = 62/434
    # 1/62 = 7/434
    # 1/434 = 1/434
    # Sum = (62+7+1)/434 = 70/434 = 35/217
    # 5/31 = 5*7/31*7 = 35/217. Yes, correct.
    
    # Can we do better? Try to find a solution with max < 434.
    # For 3 terms: 1/a + 1/b + 1/c = 5/31, with a < b < c and c < 434.
    # a >= ceil(31/5) = 7.
    
    best_solution = [7, 62, 434]
    best_max_den = 434
    
    # Search for better 3-term solutions
    for a in range(7, 200):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            break
        # Need 1/b + 1/c = r1, with b > a, c > b, c < best_max_den
        # 1/b + 1/c = r1 => c = b / (r1*b - 1)
        # r1 = p/q in lowest terms
        # c = b*q / (p*b - q)
        # For c to be integer, (p*b - q) must divide b*q
        # Also c > b means b*q/(p*b-q) > b => q/(p*b-q) > 1 => q > p*b-q => 2q > p*b => b < 2q/p
        # And c < best_max_den
        p, q = r1.numerator, r1.denominator
        # b > a, and b < 2q/p for c > b
        b_max = min(int(2 * q // p) + 1, best_max_den)
        for b in range(a + 1, b_max):
            if time.time() > deadline:
                break
            denom = p * b - q
            if denom <= 0:
                continue
            if (b * q) % denom != 0:
                continue
            c = (b * q) // denom
            if c > b and c < best_max_den and c <= 10000000:
                # Check distinctness
                if a != b and b != c and a != c:
                    best_max_den = c
                    best_solution = [a, b, c]
                    break
    
    # Also try 4-term solutions that might have smaller max
    # But 3-term with max 434 is already quite good.
    # Let's also check if there's a 2-term solution (unlikely for 5/31)
    # 1/a + 1/b = 5/31 => (a+b)/(ab) = 5/31 => 31(a+b) = 5ab
    # => 5ab - 31a - 31b = 0 => (5a-31)(5b-31) = 31*31 = 961
    # 961 = 1*961 = 31*31
    # 5a-31=1 => a=32/5 not integer
    # 5a-31=31 => a=62/5 not integer
    # So no 2-term solution.
    
    # Try to improve with more terms if time permits
    # For now, return the best 3-term solution found
    return best_solution
# EVOLVE-BLOCK-END
