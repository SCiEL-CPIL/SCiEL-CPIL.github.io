import time

def _factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factors(factors):
    primes = list(factors.keys())
    exps = [factors[p] for p in primes]
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    return divisors

def construct():
    answers = []
    deadline = time.time() + 55.0
    LIMIT_Z = 10**12

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None

        # Try divisor-based approach with x = n+1 first
        x = n + 1
        a = 4 * x - n
        b = n * x

        bf = _factorize(b)
        bf2 = {p: e * 2 for p, e in bf.items()}
        divisors_b2 = _divisors_from_factors(bf2)

        for d in divisors_b2:
            if (d + b) % a != 0:
                continue
            e_val = b * b // d
            if (e_val + b) % a != 0:
                continue
            y = (d + b) // a
            z = (e_val + b) // a
            if y <= 0 or z <= 0:
                continue
            if y > LIMIT_Z or z > LIMIT_Z:
                continue
            if y == z or y == x or z == x:
                continue
            if y > z:
                y, z = z, y
            found = [x, y, z]
            break

        if found:
            answers.append(found)
            continue

        # Fallback: brute force over x
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4

        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            B = n * x

            y_min = B // A + 1
            y_max = (2 * B - 1) // A

            if y_max < y_min:
                continue
            if y_min <= x:
                y_min = x + 1
            if y_max < y_min:
                continue

            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z <= LIMIT_Z:
                        found = [x, y, z]
                        break

            if found:
                break

        answers.append(found)

    return answers
