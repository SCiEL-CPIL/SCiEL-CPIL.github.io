import time
import math

def construct():
    deadline = time.time() + 82.0
    answers = [None] * 254
    BOUND = 10**12

    # Precompute factorizations for all n in 3..256
    def factorize(num):
        factors = {}
        temp = num
        f = 2
        while f * f <= temp:
            while temp % f == 0:
                factors[f] = factors.get(f, 0) + 1
                temp //= f
            f += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

    n_factors = {}
    for n in range(3, 257):
        n_factors[n] = factorize(n)

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None

        # 1/x + 1/y + 1/z = 4/n, x < y < z
        # x > n/4, so x_min = n//4 + 1
        # 3/x > 4/n => x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        if x_max < x_min:
            x_max = x_min

        n_fact = n_factors[n]
        n_primes = list(n_fact.keys())
        n_exps = [n_fact[p] for p in n_primes]

        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            # Need 1/y + 1/z = a/b, y > x, z > y
            # (a*y - b)(a*z - b) = b^2
            # d = a*y - b divides b^2, d > a*x - b, d < b
            # y = (d + b) / a must be integer, y > x
            # z = (b^2/d + b) / a must be integer, z > y
            b2 = b * b
            ax_minus_b = a * x - b

            # Factor b = n * x
            # Merge factorizations of n and x
            temp = x
            x_factors = {}
            f = 2
            while f * f <= temp:
                while temp % f == 0:
                    x_factors[f] = x_factors.get(f, 0) + 1
                    temp //= f
                f += 1
            if temp > 1:
                x_factors[temp] = x_factors.get(temp, 0) + 1

            # b_factors = n_factors merged with x_factors
            b_factors = dict(n_fact)
            for p, e in x_factors.items():
                b_factors[p] = b_factors.get(p, 0) + e

            # b^2 has exponents doubled
            primes = list(b_factors.keys())
            exps = [b_factors[p] * 2 for p in primes]

            # Generate divisors of b^2, but filter early
            # We need d such that:
            # 1. d < b
            # 2. d > ax_minus_b
            # 3. (d + b) % a == 0
            # 4. z = (b^2/d + b) / a is integer and z > y

            # Use recursive generation with pruning
            divisors = [1]
            for i, p in enumerate(primes):
                e = exps[i]
                current = []
                for d in divisors:
                    val = d
                    for _ in range(e + 1):
                        if val < b:  # Prune: we only need d < b
                            current.append(val)
                        val *= p
                        if val >= b:
                            break
                divisors = current

            # Sort divisors to find the first valid one quickly
            # We want d > ax_minus_b and (d + b) % a == 0
            # Filter and sort
            candidates = []
            for d in divisors:
                if d <= ax_minus_b:
                    continue
                if (d + b) % a != 0:
                    continue
                candidates.append(d)

            # Try candidates in order (smaller d gives larger z, but we just need any valid)
            for d in candidates:
                y = (d + b) // a
                if y <= x:
                    continue
                d2 = b2 // d
                if (d2 + b) % a != 0:
                    continue
                z = (d2 + b) // a
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                found = [x, y, z]
                break

            if found:
                break

        answers[idx] = found

    return answers
