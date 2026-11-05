import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    answers = [None] * 254

    # Precompute primes up to 100000 for fast factorization
    # V' can be as large as ~ n*x/gcd, with n<=256, x<=192, so V' <= 256*192 = 49152
    # sqrt(49152) ~ 222, but let's be safe with a larger sieve
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def factorize(num):
        """Factorize a number using precomputed primes."""
        if num <= 1:
            return {}
        factors = {}
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                count = 0
                while num % p == 0:
                    num //= p
                    count += 1
                factors[p] = count
        if num > 1:
            factors[num] = 1
        return factors

    def get_divisors(factors):
        """Generate all divisors from prime factorization."""
        divs = [1]
        for p, e in factors.items():
            current_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divs:
                    current_divs.append(d * power)
                power *= p
            divs = current_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            answers[idx] = None
            continue

        found = None
        # x range: n/4 < x < 3n/4, so x is in (n/4, 3n/4)
        # 1/x > 1/(3n/4) = 4/(3n) and 1/x < 1/(n/4) = 4/n
        # We need 1/x + 1/y + 1/z = 4/n
        # For distinct positive integers, x < y < z (wlog)
        # Then 3/x > 4/n => x < 3n/4
        # And 1/x < 4/n => x > n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break

            U = 4 * x - n
            V = n * x
            g = math.gcd(U, V)
            U_prime = U // g
            V_prime = V // g

            if U_prime <= 0:
                continue

            D = V_prime * V_prime

            # Factorize V_prime to get divisors of D = V_prime^2
            factors_vp = factorize(V_prime)
            factors_D = {p: 2 * e for p, e in factors_vp.items()}

            divs = get_divisors(factors_D)
            divs.sort()

            target_mod = (-V_prime) % U_prime
            sqrt_D = V_prime  # since D = V_prime^2, sqrt_D = V_prime

            for X in divs:
                if X >= sqrt_D:
                    break
                if X % U_prime != target_mod:
                    continue

                # y = (X + V_prime) / U_prime
                numerator_y = X + V_prime
                if numerator_y % U_prime != 0:
                    continue
                y = numerator_y // U_prime

                if y <= x:
                    continue
                if y > 10**12:
                    continue

                # z = (D/X + V_prime) / U_prime
                Y = D // X
                numerator_z = Y + V_prime
                if numerator_z % U_prime != 0:
                    continue
                z = numerator_z // U_prime

                if z <= y:
                    continue
                if z > 10**12:
                    continue

                # Verify: 1/x + 1/y + 1/z == 4/n
                # Cross multiply: yz*n + xz*n + xy*n == 4xyz
                # n(yz + xz + xy) == 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break

            if found:
                break

        answers[idx] = found

    return answers
