"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Try 3-term: 1/a + 1/b + 1/c = 1, a < b < c, a >= 5
    # For each a >= 5: 1/b + 1/c = 1 - 1/a = (a-1)/a
    # c = ab/(ab - a - b)
    found_3term = None
    max_a_3 = 100
    for a in range(5, max_a_3 + 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, 10000):
            if time.time() > deadline:
                break
            num = a * b
            den = a * b - a - b
            if den <= 0:
                continue
            if num % den == 0:
                c = num // den
                if c > b and c >= 5 and c <= 10000000:
                    # Verify
                    if a * b * c == b * c + a * c + a * b:
                        found_3term = [a, b, c]
                        break
        if found_3term:
            break
    
    if found_3term:
        return found_3term
    
    # Try 4-term: 1/a + 1/b + 1/c + 1/d = 1
    # For small a, b, search c, d
    found_4term = None
    max_a_4 = 15
    max_b_4 = 100
    max_c_4 = 10000
    
    for a in range(5, max_a_4 + 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, max_b_4 + 1):
            if time.time() > deadline:
                break
            rem_ab = 1 - 1/a - 1/b
            if rem_ab <= 0:
                continue
            # 1/c + 1/d = rem_ab = (ab - a - b)/(ab)
            # For each c > b: 1/d = rem_ab - 1/c
            # d = c*ab / (c*(ab-a-b) - ab)
            for c in range(b + 1, max_c_4 + 1):
                if time.time() > deadline:
                    break
                num_cd = c * a * b
                den_cd = c * (a * b - a - b) - a * b
                if den_cd <= 0:
                    continue
                if num_cd % den_cd == 0:
                    d = num_cd // den_cd
                    if d > c and d >= 5 and d <= 10000000:
                        # Verify
                        if (a * b * c * d == b * c * d + a * c * d + a * b * d + a * b * c):
                            found_4term = [a, b, c, d]
                            break
            if found_4term:
                break
        if found_4term:
            break
    
    if found_4term:
        return found_4term
    
    # Try 5-term search
    found_5term = None
    for a in range(5, 12):
        if time.time() > deadline:
            break
        for b in range(a + 1, 25):
            if time.time() > deadline:
                break
            for c in range(b + 1, 60):
                if time.time() > deadline:
                    break
                rem_abc = 1 - 1/a - 1/b - 1/c
                if rem_abc <= 0:
                    continue
                for d in range(c + 1, 200):
                    if time.time() > deadline:
                        break
                    rem_abcd = rem_abc - 1/d
                    if rem_abcd <= 0:
                        continue
                    num_e = a * b * c * d
                    den_e = num_e - b * c * d - a * c * d - a * b * d - a * b * c
                    if den_e <= 0:
                        continue
                    if num_e % den_e == 0:
                        e = num_e // den_e
                        if e > d and e >= 5 and e <= 10000000:
                            # Verify all five sum to 1
                            if (a * b * c * d * e == 
                                b * c * d * e + a * c * d * e + a * b * d * e + a * b * c * e + a * b * c * d):
                                found_5term = [a, b, c, d, e]
                                break
                if found_5term:
                    break
            if found_5term:
                break
        if found_5term:
            break
    
    if found_5term:
        return found_5term
    
    # Try 6-term search with more aggressive pruning
    found_6term = None
    for a in range(5, 10):
        if time.time() > deadline:
            break
        for b in range(a + 1, 20):
            if time.time() > deadline:
                break
            for c in range(b + 1, 40):
                if time.time() > deadline:
                    break
                rem_abc = 1 - 1/a - 1/b - 1/c
                if rem_abc <= 0:
                    continue
                for d in range(c + 1, 100):
                    if time.time() > deadline:
                        break
                    rem_abcd = rem_abc - 1/d
                    if rem_abcd <= 0:
                        continue
                    # Now need 1/e + 1/f = rem_abcd
                    # e*f = a*b*c*d*rem_abcd / (a*b*c*d*rem_abcd - e - f)
                    # Better: for each e > d: 1/f = rem_abcd - 1/e
                    # f = e / (e*rem_abcd - 1)
                    # e*rem_abcd - 1 = e*(ab - a - b - c)/(abc) - 1 = (e*(ab-a-b-c) - abc)/(abc)
                    # f = e*abc / (e*(ab-a-b-c) - abc)
                    for e in range(d + 1, 300):
                        if time.time() > deadline:
                            break
                        num_f = e * a * b * c
                        den_f = e * (a * b - a - b - c) - a * b * c
                        if den_f <= 0:
                            continue
                        if num_f % den_f == 0:
                            f = num_f // den_f
                            if f > e and f >= 5 and f <= 10000000:
                                # Verify all six sum to 1
                                prod = a * b * c * d * e * f
                                s = b*c*d*e*f + a*c*d*e*f + a*b*d*e*f + a*b*c*e*f + a*b*c*d*f + a*b*c*d*e
                                if prod == s:
                                    found_6term = [a, b, c, d, e, f]
                                    break
                    if found_6term:
                        break
                if found_6term:
                    break
            if found_6term:
                break
        if found_6term:
            break
    
    if found_6term:
        return found_6term
    
    # Fallback: use DP approach with common denominator
    target = 2520
    reached = {0: []}
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
    
    # Ultimate fallback: return a known valid solution
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/21 + 1/30 + 1/42
    # Let me verify this one more time with LCM = 210
    # 210/5=42, 210/6=35, 210/7=30, 210/10=21, 210/14=15, 210/15=14, 210/21=10, 210/30=7, 210/42=5
    # Sum = 42+35+30+21+15+14+10+7+5 = 179. Not 210.
    
    # Let me find a valid one. Try:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/18 + 1/x
    # Sum of first 8: 1/5+1/6+1/7+1/8+1/9+1/10+1/12+1/18
    # LCM = 2520
    # 2520/5=504, 2520/6=420, 2520/7=360, 2520/8=315, 2520/9=280, 2520/10=252, 2520/12=210, 2520/18=140
    # Sum = 504+420+360+315+280+252+210+140 = 2481
    # Need 2520 - 2481 = 39, so 1/x = 39/2520 = 13/840, x = 840/13. Not integer.
    
    # Try: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/18 + 1/15
    # LCM = 2520
    # 504+420+360+315+280+252+210+140+168 = 2649. Too much.
    
    # Let me just use the DP result or raise an error.
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
