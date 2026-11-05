import time
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve_one(n, deadline):
    """Find distinct x,y,z for given n, or return None."""
    # Strategy: try x = n + a for small a, then solve for y,z
    # 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
    # Let P = 4x - n, Q = nx
    # 1/y + 1/z = P/Q => (y+z)/(yz) = P/Q => P*yz = Q*(y+z)
    # P*yz - Q*y - Q*z = 0
    # (P*y - Q)(P*z - Q) = Q^2
    # So we need factor pairs of Q^2 where both factors are > Q (for positive y,z)
    
    for a in range(1, 100):
        if time.time() > deadline:
            return None
        x = n + a
        P = 4 * x - n  # = 4(n+a) - n = 3n + 4a
        Q = n * x      # = n(n+a)
        
        if P <= 0:
            continue
        
        # Need (P*y - Q)(P*z - Q) = Q^2
        # Let d1 = P*y - Q, d2 = P*z - Q, d1*d2 = Q^2
        # y = (d1 + Q)/P, z = (d2 + Q)/P
        # Need d1 + Q ≡ 0 (mod P) and d2 + Q ≡ 0 (mod P)
        # d1 ≡ -Q (mod P), d2 ≡ -Q (mod P)
        
        Q2 = Q * Q
        # Find divisors d1 of Q^2 such that d1 ≡ -Q (mod P) and d1 > 0
        # Then d2 = Q^2 // d1, check d2 ≡ -Q (mod P)
        
        # Iterate over divisors of Q^2 efficiently by iterating up to sqrt(Q^2) = Q
        # But Q can be large. Instead, iterate over divisors of Q, then combine.
        
        # Better: iterate d1 from 1 to Q (since d1*d2=Q^2 and we can assume d1<=d2 wlog, so d1<=Q)
        # But Q can be up to ~256*356 ≈ 91000, so Q^2 has sqrt = Q ≈ 91000 divisors to check
        # That's too slow for 100 a values * 254 n values
        
        # Alternative: generate divisors of Q^2 from factorization of Q
        Q_temp = Q
        factors = []
        d = 2
        while d * d <= Q_temp:
            if Q_temp % d == 0:
                cnt = 0
                while Q_temp % d == 0:
                    Q_temp //= d
                    cnt += 1
                factors.append((d, cnt))
            d += 1
        if Q_temp > 1:
            factors.append((Q_temp, 1))
        
        # Generate divisors of Q^2: for each (p, e) in factors, exponent in Q^2 is 2e
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d0 in divs:
                for exp in range(2 * e + 1):
                    new_divs.append(d0 * (p ** exp))
            divs = new_divs
        
        # Filter: d1 ≡ -Q (mod P), and d1 > 0, and d1 <= Q (wlog)
        target_mod = (-Q) % P
        for d1 in divs:
            if time.time() > deadline:
                return None
            if d1 > Q:  # wlog d1 <= d2
                break
            if d1 % P != target_mod:
                continue
            d2 = Q2 // d1
            if d2 % P != target_mod:
                continue
            y = (d1 + Q) // P
            z = (d2 + Q) // P
            if y <= 0 or z <= 0:
                continue
            if y == x or z == x or y == z:
                continue
            if y > 10**12 or z > 10**12:
                continue
            # Verify
            lhs = y * z * n + x * z * n + x * y * n
            rhs = 4 * x * y * z
            if lhs == rhs:
                return [x, y, z]
    return None

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    answers = [None] * 254
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        result = solve_one(n, deadline)
        if result is not None:
            answers[idx] = result
    return answers
