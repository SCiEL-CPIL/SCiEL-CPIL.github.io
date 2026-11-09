"""Rewritten program using algebraic factorization approach for unit fractions."""
# EVOLVE-BLOCK-START
import math

def _divisors(n):
    """Return all divisors of n sorted."""
    small = []
    large = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    large.reverse()
    return small + large

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        # Try various x values to find a valid distinct triple
        for x in range(n + 1, n + 200):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b2 = b * b
            # Factor b^2 and look for divisor d such that d*e = b^2,
            # (d+b) % a == 0 and (e+b) % a == 0
            # To keep it fast, iterate divisors of b^2
            # But b^2 can be large; instead iterate d over divisors of b^2
            # We'll generate divisors of b^2
            divs = _divisors(b2)
            for d in divs:
                e = b2 // d
                if (d + b) % a != 0:
                    continue
                if (e + b) % a != 0:
                    continue
                y = (d + b) // a
                z = (e + b) // a
                if y <= 0 or z <= 0:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                # Verify the equation
                # 1/x + 1/y + 1/z == 4/n  <=>  yz*n + xz*n + xy*n == 4xyz
                lhs = n * y * z + n * x * z + n * x * y
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
