def construct():
    from fractions import Fraction
    target = Fraction(5, 31)
    
    # Known optimal 3-term solution: 1/8 + 1/248 + 1/3720 = 5/31
    # Verify: 1/8 = 465/3720, 1/248 = 15/3720, 1/3720 = 1/3720
    # Sum = (465 + 15 + 1)/3720 = 481/3720
    # 481/3720 = 5/31?  5*3720 = 18600, 31*481 = 14911. No.
    # Let me recompute.
    # 1/8 + 1/248 + 1/3720
    # LCM of 8, 248, 3720:
    # 8 = 2^3
    # 248 = 8 * 31 = 2^3 * 31
    # 3720 = 8 * 465 = 8 * 5 * 93 = 2^3 * 5 * 3 * 31
    # LCM = 2^3 * 3 * 5 * 31 = 8 * 465 = 3720
    # 1/8 = 465/3720
    # 1/248 = 15/3720
    # 1/3720 = 1/3720
    # Sum = (465+15+1)/3720 = 481/3720
    # 481/3720: gcd(481,3720)? 481 = 13*37. 3720/13 = 286.15... not integer.
    # 3720/37 = 100.5... not integer.
    # So 481/3720 is already reduced? 481 = 13*37. 3720 = 2^3*3*5*31. No common factors.
    # 481/3720 vs 5/31: 481*31 = 14911, 5*3720 = 18600. Not equal.
    
    # The current program's fallback greedy:
    # r = 5/31 - 1/8 = (40-31)/248 = 9/248
    # d = (248+9-1)//9 = 256//9 = 28 (floor). 1/28 < 9/248? 9/248 ≈ 0.03629, 1/28 ≈ 0.03571. Yes.
    # r = 9/248 - 1/28 = (9*28 - 248)/(248*28) = (252-248)/6944 = 4/6944 = 1/1736
    # So denominators: [8, 28, 1736]. Max = 1736.
    # Check: 1/8 + 1/28 + 1/1736
    # LCM of 8,28,1736: 8=2^3, 28=4*7=2^2*7, 1736=8*217=8*7*31=2^3*7*31
    # LCM = 2^3*7*31 = 1736
    # 1/8 = 217/1736, 1/28 = 62/1736, 1/1736 = 1/1736
    # Sum = (217+62+1)/1736 = 280/1736
    # 280/1736 = 35/217 = 5/31. Yes!
    
    # So the known solution is [8, 28, 1736] with max 1736.
    # The current code searches for 3-term solutions with max < 1736 and finds none,
    # then falls back to [8, 28, 1736].
    
    # Let me search more carefully for any 3-term solution with max < 1736.
    # 1/a + 1/b + 1/c = 5/31, with a <= b <= c, max = c < 1736.
    
    best = None
    best_max = float('inf')
    
    # Search 3-term solutions
    # c = 31ab / (5ab - 31a - 31b)
    # For fixed a, b, c is determined.
    # We need 5ab - 31a - 31b > 0 and 31ab % (5ab - 31a - 31b) == 0.
    
    # Upper bound for a: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7.
    # But a can be larger. If a is too large, b and c will be large.
    # For c < 1736, we need to bound a and b.
    
    # Let's just search a from 1 to 1735, b from a+1 to 1735.
    for a in range(1, 1736):
        for b in range(a + 1, 1736):
            D = 5 * a * b - 31 * a - 31 * b
            if D <= 0:
                continue
            num = 31 * a * b
            if num % D != 0:
                continue
            c = num // D
            if c <= 0 or c > 1735:
                continue
            if c == a or c == b:
                continue
            s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
            if s != target:
                continue
            m = max(a, b, c)
            if m < best_max:
                best_max = m
                best = [a, b, c]
    
    if best is not None:
        return best
    
    # Fallback: known solution [8, 28, 1736]
    return [8, 28, 1736]
