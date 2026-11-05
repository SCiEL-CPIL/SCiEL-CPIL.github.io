import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe
    
    # Target: sum of reciprocals = 1
    # Denominators must be distinct, >= 5, <= 10000000
    # Minimize count
    
    # We know 4 terms is impossible.
    # Try to find 5 terms.
    # 1/a + 1/b + 1/c + 1/d + 1/e = 1
    
    a = 5
    # b must be > a
    for b in range(a + 1, 100):
        if time.time() > deadline:
            break
        rem_ab = Fraction(1, 1) - Fraction(1, a) - Fraction(1, b)
        if rem_ab <= 0:
            continue
        # c must be > b
        # 1/c + 1/d + 1/e = rem_ab
        # Max possible sum for c,d,e with c>b is 1/(b+1) + 1/(b+2) + 1/(b+3)
        # If rem_ab > max_sum, no solution for this b
        max_sum_cde = Fraction(1, b+1) + Fraction(1, b+2) + Fraction(1, b+3)
        if rem_ab > max_sum_cde:
            continue
        
        for c in range(b + 1, 1000):
            if time.time() > deadline:
                break
            rem_abc = rem_ab - Fraction(1, c)
            if rem_abc <= 0:
                break
            # 1/d + 1/e = rem_abc
            # d > c
            # 1/d < rem_abc => d > 1/rem_abc
            # Also 1/d > rem_abc / 2 (since 1/e < 1/d) => d < 2/rem_abc
            d_min = c + 1
            d_max = int(2 / rem_abc) + 2
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    break
                rem_abcd = rem_abc - Fraction(1, d)
                if rem_abcd <= 0:
                    break
                # 1/e = rem_abcd
                # e = 1 / rem_abcd
                if rem_abcd.denominator % rem_abcd.numerator == 0:
                    e = rem_abcd.denominator // rem_abcd.numerator
                    if e > d and e <= 10000000:
                        return [a, b, c, d, e]
                # If rem_abcd is not unit fraction, no solution for this d
    
    # Fallback: return a known valid solution with more terms if 5 not found quickly
    # Or just use the DP result if it's fast enough, but 5 is better.
    # A known 6-term solution: 5, 6, 7, 8, 9, 10? No sum < 1.
    # 5, 6, 7, 8, 9, 12?
    # 1/5+1/6+1/7+1/8+1/9+1/12 = 0.2+0.166+0.142+0.125+0.111+0.083 = 0.827
    
    # Use the original DP logic as fallback if search fails
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no baseline representation")
