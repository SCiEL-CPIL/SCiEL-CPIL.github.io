"""Optimized constructor for distinct three-term unit-fraction reps of 4/n."""
import time
import math
from math import gcd

DENOM_BOUND = 10**12

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        x_min = n // 4 + 1
        found = None
        
        # Try x values: near x_min first (most likely to have solutions)
        # Then expand outward
        x_tried = set()
        
        # Phase 1: x from x_min to x_min + 50
        for offset in range(0, 51):
            x = x_min + offset
            x_tried.add(x)
            found = _try_x(n, x, deadline, start_time)
            if found:
                break
            if time.time() > deadline:
                break
        
        # Phase 2: if not found, try more x values
        if not found:
            for offset in [52, 55, 60, 70, 80, 100, 120, 150, 200, 250, 300, 400, 500]:
                x = x_min + offset
                if x in x_tried:
                    continue
                x_tried.add(x)
                found = _try_x(n, x, deadline, start_time)
                if found:
                    break
                if time.time() > deadline:
                    break
        
        # Phase 3: try some larger x values
        if not found:
            for offset in [600, 800, 1000, 1500, 2000, 3000, 5000]:
                x = x_min + offset
                if x in x_tried:
                    continue
                x_tried.add(x)
                found = _try_x(n, x, deadline, start_time)
                if found:
                    break
                if time.time() > deadline:
                    break
        
        answers[i] = found
    
    return answers


def _try_x(n, x, deadline, start_time):
    """Try to find y, z given n and x such that 1/x + 1/y + 1/z = 4/n."""
    A = 4 * x - n
    if A <= 0:
        return None
    B = n * x
    
    # y must satisfy: B/A < y <= 2B/A and y > x
    y_lo = B // A + 1
    y_hi = (2 * B) // A
    if y_hi < y_lo:
        return None
    if y_lo <= x:
        y_lo = x + 1
    if y_lo > y_hi:
        return None
    
    g = gcd(A, B)
    a_prime = A // g
    b_prime = B // g
    
    # y = g * m, z = b_prime * m / (m * a_prime - b_prime)
    # Need: m * a_prime - b_prime > 0, divides b_prime * m
    # m > b_prime / a_prime
    m_min = b_prime // a_prime + 1
    if m_min < 1:
        m_min = 1
    
    y_min_check = max(y_lo, x + 1)
    m_lo = (y_min_check + g - 1) // g  # ceil division
    if m_lo < m_min:
        m_lo = m_min
    m_hi = y_hi // g
    if m_hi < m_lo:
        return None
    
    # Strategy: for large ranges, use divisor enumeration on b_prime
    # (m*a' - b') | b'*m  =>  let d = m*a' - b', then d | b'*m
    # d = m*a' - b' > 0, and m = (d + b')/a' must be integer
    # So d ≡ -b' (mod a'), i.e., d = k*a' - b' for some k >= 1
    # Then m = k, and d = k*a' - b'
    # Need: (k*a' - b') | b'*k
    # i.e., b'*k mod (k*a' - b') == 0
    
    # Alternative: enumerate divisors of b_prime * m for candidate m values
    # But m can be large. Instead, note that if d | b'*m and d = m*a' - b',
    # then d | b'*m and d | m*a' - b', so d | b'*m + b'*(m*a'-b')/a'... 
    # Simpler: just scan m in a smart way.
    
    range_len = m_hi - m_lo + 1
    
    if range_len <= 20000:
        # Direct scan
        for m in range(m_lo, m_hi + 1):
            if (m - m_lo) % 5000 == 0 and time.time() > deadline:
                return None
            denom_z = m * a_prime - b_prime
            if denom_z <= 0:
                continue
            num_z = b_prime * m
            if num_z % denom_z != 0:
                continue
            z_val = num_z // denom_z
            y_val = g * m
            if z_val <= y_val:
                continue
            if z_val > DENOM_BOUND:
                continue
            if x != y_val and y_val != z_val and x != z_val:
                return [x, y_val, z_val]
    else:
        # For large ranges, try to find solutions using divisor structure
        # We need (m*a' - b') | b'*m
        # Let's try: enumerate small values of (m*a' - b') that divide b'*m
        # d = m*a' - b', so m = (d + b')/a'
        # d must be positive, d ≡ -b' (mod a'), and d | b'*(d+b')/a'
        # Since a' and b' are coprime, d | b'*(d+b')/a' means a'*d | b'*(d+b')
        # => a'*d | b'*d + b'^2 => since d | b'*d, need a'*d | b'^2
        # Wait, that's not right. Let me redo.
        # a'*d | b'*(d+b') = b'*d + b'^2
        # d | b'*d always, so we need: a'*d | b'^2 + b'*d - b'*d = b'^2? No.
        # a'*d | b'*(d+b') means there exists k: b'*(d+b') = k*a'*d
        # => b'*d + b'^2 = k*a'*d
        # => b'^2 = d*(k*a' - b')
        # So d must divide b'^2!
        # And k = (b'*(d+b'))/(a'*d) must be integer, which is equivalent to a'*d | b'*(d+b')
        # But from b'^2 = d*(k*a' - b'), we get k*a' - b' = b'^2/d, so k = (b'^2/d + b')/a'
        # For k to be integer: a' | (b'^2/d + b')
        
        # So: enumerate divisors d of b'^2 such that:
        # 1. d > 0
        # 2. m = (d + b')/a' is integer (i.e., d ≡ -b' mod a')
        # 3. m >= m_lo and m <= m_hi
        # 4. z = b'*m/d = b'*(d+b')/(a'*d) is integer > y = g*m
        # 5. All distinct and within bound
        
        # Factor b_prime to enumerate divisors
        bp = b_prime
        if bp > 10**9:
            # Too large for full factorization; fall back to limited scan
            scan_limit = min(m_hi, m_lo + 50000)
            for m in range(m_lo, scan_limit + 1):
                if (m - m_lo) % 10000 == 0 and time.time() > deadline:
                    return None
                denom_z = m * a_prime - b_prime
                if denom_z <= 0:
                    continue
                num_z = b_prime * m
                if num_z % denom_z != 0:
                    continue
                z_val = num_z // denom_z
                y_val = g * m
                if z_val <= y_val:
                    continue
                if z_val > DENOM_BOUND:
                    continue
                if x != y_val and y_val != z_val and x != z_val:
                    return [x, y_val, z_val]
            return None
        
        # Factor bp
        factors = _factorize(bp)
        divisors = _get_divisors(factors)
        
        for d in divisors:
            if d <= 0:
                continue
            # Check d ≡ -b' (mod a')
            if (d + b_prime) % a_prime != 0:
                continue
            m_val = (d + b_prime) // a_prime
            if m_val < m_lo or m_val > m_hi:
                continue
            y_val = g * m_val
            if y_val <= x:
                continue
            if y_val > y_hi:
                continue
            # z = b_prime * m_val / d
            if (b_prime * m_val) % d != 0:
                continue
            z_val = (b_prime * m_val) // d
            if z_val <= y_val:
                continue
            if z_val > DENOM_BOUND:
                continue
            if x != y_val and y_val != z_val and x != z_val:
                return [x, y_val, z_val]
    
    return None


def _factorize(num):
    """Return list of (prime, exponent) for num."""
    factors = []
    d = 2
    while d * d <= num:
        if num % d == 0:
            exp = 0
            while num % d == 0:
                num //= d
                exp += 1
            factors.append((d, exp))
        d += 1 if d == 2 else 2
    if num > 1:
        factors.append((num, 1))
    return factors


def _get_divisors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    return divisors
