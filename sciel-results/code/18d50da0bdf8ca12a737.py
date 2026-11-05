import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    answers = [None] * 254

    # Precompute divisors of k^2 for all k up to max possible V_prime
    # V_prime = V/gcd(U,V) where U=4x-n, V=n*x
    # Max V = n*x <= 256*192 = 49152, so V_prime <= 49152
    max_vp = 50000
    # Precompute smallest prime factor for fast factorization
    spf = list(range(max_vp + 1))
    for i in range(2, int(max_vp**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_vp + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute divisors of k^2 for each k
    divisors_sq = [None] * (max_vp + 1)
    divisors_sq[0] = []
    divisors_sq[1] = [1]
    for k in range(2, max_vp + 1):
        # Factorize k using spf
        factors = {}
        n_val = k
        while n_val > 1:
            p = spf[n_val]
            cnt = 0
            while n_val % p == 0:
                n_val //= p
                cnt += 1
            factors[p] = cnt
        # Generate divisors of k^2
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            power = 1
            for _ in range(2 * e + 1):
                for d in divs:
                    new_divs.append(d * power)
                power *= p
            divs = new_divs
        divs.sort()
        divisors_sq[k] = divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            answers[idx] = None
            continue

        found = None
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

            if V_prime > max_vp:
                continue

            D = V_prime * V_prime
            divs = divisors_sq[V_prime]
            sqrt_D = V_prime

            target_mod = (-V_prime) % U_prime

            for X in divs:
                if X >= sqrt_D:
                    break
                if X % U_prime != target_mod:
                    continue

                numerator_y = X + V_prime
                if numerator_y % U_prime != 0:
                    continue
                y = numerator_y // U_prime

                if y <= x:
                    continue
                if y > 10**12:
                    continue

                Y = D // X
                numerator_z = Y + V_prime
                if numerator_z % U_prime != 0:
                    continue
                z = numerator_z // U_prime

                if z <= y:
                    continue
                if z > 10**12:
                    continue

                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break

            if found:
                break

        answers[idx] = found

    return answers
