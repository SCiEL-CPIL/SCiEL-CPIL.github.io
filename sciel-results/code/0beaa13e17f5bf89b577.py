import time
import math
from math import gcd

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # Sieve for primes up to 2000 (need to factor numbers up to ~256*256)
    limit = 2000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]

    def factorize(num):
        """Returns dict of prime factors and exponents for num."""
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors[p] = count
        if temp > 1:
            factors[temp] = 1
        return factors

    def merge_factors(f1, f2):
        """Merge two factorization dicts."""
        merged = dict(f1)
        for p, e in f2.items():
            if p in merged:
                merged[p] += e
            else:
                merged[p] = e
        return merged

    def get_divisors_from_factors(factors):
        """Generates all divisors from a prime factorization dict."""
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors

    def find_solution_for_n(n):
        """Find distinct x < y < z with 1/x + 1/y + 1/z = 4/n."""
        # Precompute factorization of n
        n_factors = factorize(n)
        
        # x must satisfy: x > n/4 and x < 3n/4
        min_x = n // 4 + 1
        max_x = (3 * n) // 4 - 1

        for x in range(min_x, max_x + 1):
            A = 4 * x - n
            B = n * x
            B2 = B * B

            # We need u * v = B2 with u < B, v > B
            # y = (u+B)/A, z = (v+B)/A must be integers > x
            # y > x => u > 4x^2 - 2nx
            lower_bound_u = 4 * x * x - 2 * n * x
            if lower_bound_u < 0:
                lower_bound_u = 0

            # Factor B = n * x using precomputed n_factors
            x_factors = factorize(x)
            b_factors = merge_factors(n_factors, x_factors)
            b2_factors = {p: 2 * e for p, e in b_factors.items()}

            divisors = get_divisors_from_factors(b2_factors)

            for u in divisors:
                if u >= B:
                    continue
                if u <= lower_bound_u:
                    continue
                if (u + B) % A != 0:
                    continue

                v = B2 // u
                if v <= u:
                    continue
                if (v + B) % A != 0:
                    continue

                y = (u + B) // A
                z = (v + B) // A

                if y > x and z > y and z <= 1000000000000:
                    return [x, y, z]

        return None

    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = find_solution_for_n(n)
        answers.append(found)

    return answers
