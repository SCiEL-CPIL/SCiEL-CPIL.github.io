import time
import math
from math import gcd

def construct():
    deadline = time.time() + 78.0
    answers = [None] * 254
    BOUND = 10**12

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break

        found = None

        # We want distinct positive integers x < y < z with 1/x + 1/y + 1/z = 4/n
        # x must satisfy n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = min(3 * n // 4, BOUND)

        # Try smaller x values first (they tend to give smaller y, z)
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break

            a = 4 * x - n  # positive
            b = n * x

            # 1/y + 1/z = a/b
            # y must satisfy: 2b/a < y < b (since z > y implies 1/y + 1/z < 2/y, so a/b < 2/y => y < 2b/a... wait)
            # Actually: 1/y + 1/z = a/b with y < z
            # 2/y > a/b => y < 2b/a
            # 1/y < a/b => y > b/a
            # So b/a < y < 2b/a
            y_min = b // a + 1
            y_max = min(2 * b // a - 1, BOUND)

            if y_min > y_max:
                continue

            # For each y, z = ab / (ay - b)
            # We need (ay - b) to divide ab, and z > y, z <= BOUND
            # Let d = ay - b, so d divides ab and y = (d+b)/a, z = ab/d
            # d must be in range: since y > y_min, d = ay - b > a*y_min - b = a*(b//a+1) - b >= b + a - b = a (roughly)
            # More precisely: d = ay - b, with y in [y_min, y_max]
            # d_min = a*y_min - b, d_max = a*y_max - b

            # z = ab/d, we need z > y = (d+b)/a
            # ab/d > (d+b)/a => a^2*b > d*(d+b) => d^2 + bd - a^2*b < 0
            # d < (-b + sqrt(b^2 + 4a^2*b))/2

            # Also z <= BOUND => ab/d <= BOUND => d >= ab/BOUND
            # And d must divide ab

            # Let's iterate over divisors of ab that are in the valid range for d
            # ab can be up to ~256 * 192 * 256 * 192 ≈ 2.4e9, manageable

            # Factor ab
            temp = a * b
            factors = {}
            f = 2
            while f * f <= temp:
                while temp % f == 0:
                    factors[f] = factors.get(f, 0) + 1
                    temp //= f
                f += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1

            # Generate divisors of ab
            divisors = [1]
            for p, exp in factors.items():
                new_divs = []
                for d in divisors:
                    val = d
                    for i in range(exp + 1):
                        new_divs.append(val)
                        val *= p
                divisors = new_divs

            # For each divisor d of ab, check if it gives valid y, z
            ab = a * b
            for d in divisors:
                # d = ay - b, so y = (d + b) / a must be integer
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y <= x or y > y_max or y < y_min:
                    continue
                # z = ab / d
                if ab % d != 0:
                    continue
                z = ab // d
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                # Check: x, y, z all distinct and positive
                if x == y or y == z or x == z:
                    continue
                found = [x, y, z]
                break

            if found:
                break

        answers[idx] = found

    return answers
