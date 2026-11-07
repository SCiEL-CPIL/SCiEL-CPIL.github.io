import time
import math
import sys
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def check_time():
        return time.time() > deadline

    def sieve_sigma(n):
        """Compute sigma(k) for all k in [1, n] using a sieve."""
        sigma = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(i, n + 1, i):
                sigma[j] += i
        return sigma

    def small_primes(n):
        if n < 2:
            return []
        is_p = bytearray(b'\x01') * (n + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(n + 1) if is_p[i]]

    max_m = 10**18

    # Strategy: sieve sigma up to a large limit, collect M = k*sigma(k), count collisions
    # We'll do multiple passes with different limits if time permits

    # Phase 1: Sieve up to 2,000,000
    limit1 = 2_000_000
    sigma_arr = sieve_sigma(limit1)
    m_counts = defaultdict(int)
    m_examples = defaultdict(list)

    for k in range(2, limit1 + 1):
        m_val = k * sigma_arr[k]
        if m_val <= max_m:
            m_counts[m_val] += 1
            if len(m_examples[m_val]) < 200:
                m_examples[m_val].append(k)

    # Phase 2: Sieve up to 5,000,000 (extends the range)
    if not check_time():
        limit2 = 5_000_000
        sigma_arr2 = sieve_sigma(limit2)
        for k in range(limit1 + 1, limit2 + 1):
            m_val = k * sigma_arr2[k]
            if m_val <= max_m:
                m_counts[m_val] += 1
                if len(m_examples[m_val]) < 200:
                    m_examples[m_val].append(k)

    # Phase 3: Sieve up to 10,000,000
    if not check_time():
        limit3 = 10_000_000
        sigma_arr3 = sieve_sigma(limit3)
        for k in range(limit2 + 1, limit3 + 1):
            m_val = k * sigma_arr3[k]
            if m_val <= max_m:
                m_counts[m_val] += 1
                if len(m_examples[m_val]) < 200:
                    m_examples[m_val].append(k)

    # Phase 4: Sieve up to 20,000,000
    if not check_time():
        limit4 = 20_000_000
        sigma_arr4 = sieve_sigma(limit4)
        for k in range(limit3 + 1, limit4 + 1):
            m_val = k * sigma_arr4[k]
            if m_val <= max_m:
                m_counts[m_val] += 1
                if len(m_examples[m_val]) < 200:
                    m_examples[m_val].append(k)

    # Phase 5: Sieve up to 50,000,000
    if not check_time():
        limit5 = 50_000_000
        sigma_arr5 = sieve_sigma(limit5)
        for k in range(limit4 + 1, limit5 + 1):
            m_val = k * sigma_arr5[k]
            if m_val <= max_m:
                m_counts[m_val] += 1
                if len(m_examples[m_val]) < 200:
                    m_examples[m_val].append(k)

    if not m_counts:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Sort by count descending
    sorted_ms = sorted(m_counts.items(), key=lambda item: item[1], reverse=True)

    # Factorization helpers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        c = 1
        while True:
            x = 2
            y = 2
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            c += 1

    def factorize(n):
        factors = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
        else:
            stack = [n]
            while stack:
                x = stack.pop()
                if x == 1:
                    continue
                if is_prime(x):
                    factors.append(x)
                else:
                    d = pollard_rho(x)
                    stack.append(d)
                    stack.append(x // d)
        return sorted(factors)

    def generate_divisors_from_factors(factors):
        c = Counter(factors)
        primes_list = list(c.keys())
        exponents_list = [c[p] for p in primes_list]
        divs = [1]
        for i in range(len(primes_list)):
            p = primes_list[i]
            e = exponents_list[i]
            current_divs = []
            power = 1
            for k in range(e + 1):
                for d in divs:
                    current_divs.append(d * power)
                power *= p
            divs = current_divs
        return divs

    # For top candidates, find ALL valid k by checking divisors of M
    best_m = 0
    best_k_list = []

    num_candidates_to_check = min(30, len(sorted_ms))

    for idx in range(num_candidates_to_check):
        if check_time():
            break

        M, count = sorted_ms[idx]
        sqrt_M = math.isqrt(M)

        try:
            factors = factorize(M)
            if not factors:
                continue

            divs = generate_divisors_from_factors(factors)
            m_primes_set = set(factors)

            valid_ks = []
            for d in divs:
                if d > sqrt_M:
                    continue
                if d < 2:
                    continue

                temp_d = d
                sigma_d = 1
                for p in m_primes_set:
                    if temp_d % p == 0:
                        e = 0
                        while temp_d % p == 0:
                            temp_d //= p
                            e += 1
                        sigma_d *= (p**(e + 1) - 1) // (p - 1)

                if temp_d != 1:
                    continue

                if d * sigma_d == M:
                    valid_ks.append(d)

            if len(valid_ks) > len(best_k_list):
                best_k_list = sorted(set(valid_ks))
                best_m = M

        except Exception:
            continue

    if best_k_list:
        sqrt_m = math.isqrt(best_m)
        valid_k = [k for k in best_k_list if k <= sqrt_m and k >= 2]
        valid_k.sort()
        if valid_k:
            return {'M': int(best_m), 'k': [int(k) for k in valid_k]}

    # Fallback: use collected examples
    if sorted_ms:
        best_m, best_count = sorted_ms[0]
        examples = m_examples.get(best_m, [])
        sqrt_m = math.isqrt(best_m)
        valid_k = sorted(set(k for k in examples if k <= sqrt_m and k >= 2))
        if valid_k:
            return {'M': int(best_m), 'k': [int(k) for k in valid_k]}

    return {'M': 333312, 'k': [336, 372, 434]}

if __name__ == "__main__":
    result = construct()
    print(result)
