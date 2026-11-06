import time

def construct():
    deadline = time.time() + 85.0
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        found = None
        x_min = n // 4 + 1  # smallest x such that 4x > n
        # Try x values starting from x_min
        for x in range(x_min, x_min + 200):
            if time.time() > deadline:
                break
            a = 4 * x - n  # must be positive
            if a <= 0:
                continue
            b = n * x
            # Need (ay-b)(az-b) = b^2
            # Let d = ay - b, then d | b^2, y = (d+b)//a, z = (b^2//d + b)//a
            # y > x, z > y, all distinct
            b2 = b * b
            # Find divisors d of b^2 such that d > b (so y > b/a = n/4... need y > x)
            # Actually y = (d+b)/a > x means d+b > ax = a*x, so d > ax - b = a*x - n*x = x(a-n) = x(4x-n-n) = x(4x-2n)
            # This is getting complex. Let's just iterate divisors.
            # For efficiency, iterate d over divisors of b^2
            # b^2 can be large, but for n<=256, x is small, so b is manageable
            # Let's find all divisors of b^2
            # b = n*x, for n<=256, x up to ~250, b up to ~64000, b^2 up to ~4e9
            # Finding divisors of b^2 by trial division up to sqrt(b^2) = b ~ 64000 is feasible
            # But we need to be careful about time
            # Alternative: iterate y from x+1 upward and check if z is integer
            # 1/z = a/b - 1/y = (ay - b)/(by), so z = by/(ay-b)
            # Need ay > b, so y > b/a
            y_min = max(x + 1, b // a + 1)
            # z = by/(ay-b) must be integer and > y
            # ay - b must divide by
            # Iterate y
            # Upper bound: as y increases, ay-b increases, z = by/(ay-b) decreases
            # z > y means by/(ay-b) > y, so b > y(ay-b) = ay^2 - by, so ay^2 - by - b < 0
            # y < (b + sqrt(b^2 + 4ab))/2a approximately
            # For large y, z approaches b/a from above
            # Let's set a reasonable upper bound
            # z > y and z = by/(ay-b), so by > y(ay-b), b > ay^2 - by, ay^2 - by - b < 0
            # y^2 - (b/a)y - b/a < 0, y < (b/a + sqrt((b/a)^2 + 4b/a))/2
            # This can be large. Let's just iterate and break when z <= y
            for y in range(y_min, y_min + 100000):
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * y) % denom != 0:
                    continue
                z = (b * y) // denom
                if z > y and z != x and z != y and y != x:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    return answers
