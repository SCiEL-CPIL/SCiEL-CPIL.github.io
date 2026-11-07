import time
import math
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    max_m = 10**18

    # Sieve for primes up to a reasonable limit
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Use primes up to 200 for smooth number generation
    primes_gen = sieve(200)
    n_primes = len(primes_gen)

    # Iterative smooth number generation using stack
    m_counts = defaultdict(int)
    m_examples = defaultdict(list)

    # Stack entries: (idx, current_k, current_sigma)
    stack = [(0, 1, 1)]
    while stack:
        if time.time() > deadline - 40:
            break
        idx, current_k, current_sigma = stack.pop()
        if idx == n_primes:
            if current_k > 1:
                m_val = current_k * current_sigma
                if m_val <= max_m:
                    m_counts[m_val] += 1
                    if len(m_examples[m_val]) < 10:
                        m_examples[m_val].append(current_k)
            continue
        p = primes_gen[idx]
        # Option: exponent 0
        stack.append((idx + 1, current_k, current_sigma))
        # Option: exponent e >= 1
        k_curr = current_k
        sum_powers = 1
        p_power = 1
        while True:
            p_power *= p
            k_curr *= p
            sum_powers += p_power
            sigma_curr_new = current_sigma * sum_powers
            if k_curr * sigma_curr_new > max_m:
                break
            stack.append((idx + 1, k_curr, sigma_curr_new))

    if not m_counts:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Sort by count, take top candidates
    sorted_ms = sorted(m_counts.items(), key=lambda item: item[1], reverse=True)

    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Small primes check
        for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        # Miller-Rabin
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses for n < 3,317,044,064,679,887,385,961,981
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        if n % 7 == 0:
            return 7
        if n % 11 == 0:
            return 11
        if n % 13 == 0:
            return 13
        while True:
            x = 2
            y = 2
            d = 1
            c = 1
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
        primes_list = sorted(c.keys())
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

    def find_all_k_for_M(M):
        if M < 2:
            return []

        factors = factorize(M)
        divs = generate_divisors_from_factors(factors)

        m_primes = sorted(set(factors))
        valid_ks = []
        sqrt_m = math.isqrt(M)

        for d in divs:
            if d > sqrt_m:
                continue
            if d < 1:
                continue

            temp_d = d
            sigma_d = 1
            for p in m_primes:
                if temp_d % p == 0:
                    e = 0
                    while temp_d % p == 0:
                        temp_d //= p
                        e += 1
                    sigma_d *= (p ** (e + 1) - 1) // (p - 1)
            if temp_d > 1:
                temp_factors = factorize(temp_d)
                temp_c = Counter(temp_factors)
                for p, e in temp_c.items():
                    sigma_d *= (p ** (e + 1) - 1) // (p - 1)

            if d * sigma_d == M:
                valid_ks.append(d)

        return valid_ks

    # Try top candidates to find all k's
    best_m = 0
    best_k_list = []

    num_candidates = min(80, len(sorted_ms))

    for i in range(num_candidates):
        if time.time() > deadline - 5:
            break

        M, count = sorted_ms[i]

        try:
            valid_ks = find_all_k_for_M(M)
            if len(valid_ks) > len(best_k_list):
                best_k_list = valid_ks
                best_m = M
        except Exception:
            continue

    # If we didn't find anything better, use the smooth examples
    if not best_k_list:
        best_m, count = sorted_ms[0]
        best_k_list = m_examples[best_m]

    # Final validation
    distinct_k = list(set(best_k_list))
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in distinct_k if 1 <= k <= sqrt_m]
    valid_k.sort()

    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]

    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
