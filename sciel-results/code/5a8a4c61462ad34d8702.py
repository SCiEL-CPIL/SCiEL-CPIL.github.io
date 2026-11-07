import time
import math
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
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

    # Use primes up to 500 for smooth number generation
    primes_gen = sieve(500)

    # Generate smooth numbers k and compute M = k * sigma(k)
    # Group by M to find high-collision values
    m_counts = defaultdict(int)
    m_examples = defaultdict(list)

    # Iterative smooth number generation to avoid recursion depth issues
    # and to better manage time budget
    # We use a BFS/DFS hybrid with pruning

    # Precompute log bounds for pruning: log(k) + log(sigma(k)) <= log(max_m)
    log_max_m = math.log(max_m)

    # Use iterative approach with explicit stack
    # State: (idx, current_k, current_sigma, log_k_sigma)
    # We track log to prune early

    def gen_smooth_iterative():
        # Stack of (idx, current_k, current_sigma)
        # We'll use a recursive-like approach but with time checks
        # For deeper prime sets, we need to be smarter

        # Strategy: generate all smooth numbers up to sqrt(max_m) roughly
        # but track their k*sigma(k) values

        # Use iterative deepening on number of primes used
        # First, generate with first N primes, then add more

        # For efficiency, let's do a two-pass approach:
        # Pass 1: Generate with smaller prime set to get initial candidates
        # Pass 2: Refine with larger prime set if time permits

        # Actually, let's just do a single pass with the full prime set
        # but with aggressive pruning

        # Recursive with time check
        def _gen(idx, cur_k, cur_sigma):
            if time.time() > deadline - 40:
                return

            if idx == len(primes_gen):
                if cur_k > 1:
                    m_val = cur_k * cur_sigma
                    if m_val <= max_m:
                        m_counts[m_val] += 1
                        if len(m_examples[m_val]) < 20:
                            m_examples[m_val].append(cur_k)
                return

            p = primes_gen[idx]

            # Option: exponent 0
            _gen(idx + 1, cur_k, cur_sigma)

            # Option: exponent e >= 1
            k_curr = cur_k
            sum_powers = 1
            p_power = 1

            # Pruning: k_curr * cur_sigma * sum_powers <= max_m
            # => k_curr * sum_powers <= max_m / cur_sigma
            limit = max_m // cur_sigma if cur_sigma > 0 else max_m

            while True:
                p_power *= p
                k_curr *= p
                sum_powers += p_power

                if k_curr > limit // sum_powers:
                    break

                _gen(idx + 1, k_curr, cur_sigma * sum_powers)

        _gen(0, 1, 1)

    gen_smooth_iterative()

    if not m_counts:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Sort by count, take top candidates
    sorted_ms = sorted(m_counts.items(), key=lambda item: item[1], reverse=True)

    # Pollard Rho for factoring
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
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
        if n % 5 == 0:
            return 5
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

    # Sort by count descending, but also consider M size (smaller M may be easier to factor)
    num_candidates = min(50, len(sorted_ms))

    for i in range(num_candidates):
        if time.time() > deadline - 10:
            break

        M, count = sorted_ms[i]

        # Skip if we already have at least as many solutions
        if count <= len(best_k_list):
            break

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

    # Ensure we have at least one solution
    if not valid_k:
        # Fallback: try to find any valid k for best_m
        for M, count in sorted_ms[:5]:
            try:
                valid_ks = find_all_k_for_M(M)
                if valid_ks:
                    best_m = M
                    best_k_list = valid_ks
                    break
            except Exception:
                continue
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
