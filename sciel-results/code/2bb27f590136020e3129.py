"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import time
import math

def _divisors_of(n):
    """Return all divisors of n in sorted order."""
    if n <= 0:
        return []
    small = []
    large = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]

def construct():
    deadline = time.time() + 82  # safety margin before the 90s kill
    answers = []

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        # x is the smallest denominator: 1/x < 4/n => x > n/4, and 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        # Try x values in order of increasing a (=4x-n), i.e., from x_min upward
        # But for efficiency, try x where the range of m is smaller first
        # m range is ~ b/a = nx/(4x-n), which decreases as x increases
        # So try larger x first for faster search
        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
            if found:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0 or b <= 0:
                continue
            
            b2 = b * b
            
            # d = a*m - b, need 1 <= d < b and d | b^2
            # For each divisor d of b^2 with 1 <= d < b:
            #   m = (d + b) / a must be integer and m > x
            #   y = (d + b) / a = m
            #   z = (b^2/d + b) / a
            # We need m > x, y > x, z > y, all distinct, all <= 10^12
            
            # Get divisors of b^2 that are in range [1, b-1]
            # For efficiency, only check divisors d where (d+b) % a == 0
            # and m = (d+b)//a > x
            
            # Iterate over divisors of b^2
            # But b^2 can be large; let's use the divisor enumeration
            # For small b, enumerate all divisors. For large b, this could be slow.
            # Alternative: iterate m in range and check b^2 % (a*m - b) == 0
            
            # Let's try the divisor approach for now
            # But we need to be careful: b^2 can be up to (256*256)^2 ≈ 4.3 * 10^8
            # Number of divisors is at most ~1000-2000 for such numbers
            
            if b2 > 10**14:
                # Fallback to m-range search for very large b^2
                m_min = max(x + 1, (b + 1 + a - 1) // a)
                m_max = (2 * b - 1) // a
                if m_min > m_max:
                    continue
                for m in range(m_min, m_max + 1):
                    if time.time() > deadline:
                        break
                    if found:
                        break
                    d = a * m - b
                    if d <= 0 or d >= b:
                        continue
                    if b2 % d != 0:
                        continue
                    d2 = b2 // d
                    num_y = d + b
                    if num_y % a != 0:
                        continue
                    y = num_y // a
                    if y <= x:
                        continue
                    num_z = d2 + b
                    if num_z % a != 0:
                        continue
                    z = num_z // a
                    if z <= y:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    if z > 10**12 or y > 10**12 or x > 10**12:
                        continue
                    lhs_num = n * y * z + n * x * z + n * x * y
                    rhs_num = 4 * x * y * z
                    if lhs_num == rhs_num:
                        found = [x, y, z]
                        break
            else:
                # Use divisor enumeration
                divs = _divisors_of(b2)
                for d in divs:
                    if time.time() > deadline:
                        break
                    if found:
                        break
                    if d <= 0 or d >= b:
                        continue
                    num_y = d + b
                    if num_y % a != 0:
                        continue
                    y = num_y // a
                    if y <= x:
                        continue
                    d2 = b2 // d
                    num_z = d2 + b
                    if num_z % a != 0:
                        continue
                    z = num_z // a
                    if z <= y:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    if z > 10**12 or y > 10**12 or x > 10**12:
                        continue
                    lhs_num = n * y * z + n * x * z + n * x * y
                    rhs_num = 4 * x * y * z
                    if lhs_num == rhs_num:
                        found = [x, y, z]
                        break
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
