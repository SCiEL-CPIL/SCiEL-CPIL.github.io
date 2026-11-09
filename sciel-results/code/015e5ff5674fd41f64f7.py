"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before the 90s kill
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
            # d = a*m - b, need 1 <= d < b
            # m >= ceil((b+1)/a), m <= floor((2b-1)/a)
            m_min = (b + 1 + a - 1) // a  # ceil((b+1)/a)
            m_max = (2 * b - 1) // a  # floor((2b-1)/a)
            if m_min > m_max:
                continue
            # Also need y > x: (d+b)/a > x => d > a*x - b
            # d = a*m - b > a*x - b => a*m > a*x => m > x
            # So m >= x + 1
            if m_min <= x:
                m_min = x + 1
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
                # Check d | b^2
                b2 = b * b
                if b2 % d != 0:
                    continue
                d2 = b2 // d
                # y = (d + b) / a, z = (d2 + b) / a
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
                # Verify via cross multiplication
                lhs_num = n * y * z + n * x * z + n * x * y
                rhs_num = 4 * x * y * z
                if lhs_num == rhs_num:
                    found = [x, y, z]
                    break
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
