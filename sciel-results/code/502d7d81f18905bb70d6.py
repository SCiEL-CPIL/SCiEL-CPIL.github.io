import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    def time_ok():
        return time.time() - start_time < DEADLINE

    # 1. Sieve of Eratosthenes for primes up to 10^7 using NumPy
    LIMIT = 10**7
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = np.nonzero(is_prime)[0].tolist()

    # 2. Precompute phi up to LIMIT using NumPy
    phi = np.arange(LIMIT + 1, dtype=np.int64)
    for p in primes:
        if p > LIMIT:
            break
        phi[p::p] = phi[p::p] - phi[p::p] // p

    # 3. Fast phi for n > LIMIT using Pollard's Rho
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factor(n, factors=None):
        if factors is None:
            factors = {}
        if n == 1:
            return factors
        if n <= LIMIT:
            # Trial division with precomputed primes
            for p in primes:
                if p * p > n:
                    break
                while n % p == 0:
                    factors[p] = factors.get(p, 0) + 1
                    n //= p
            if n > 1:
                factors[n] = factors.get(n, 0) + 1
            return factors
        # For larger n, use Pollard's Rho
        while n > 1:
            if n <= LIMIT:
                for p in primes:
                    if p * p > n:
                        break
                    while n % p == 0:
                        factors[p] = factors.get(p, 0) + 1
                        n //= p
                if n > 1:
                    factors[n] = factors.get(n, 0) + 1
                break
            d = pollard_rho(n)
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
        return factors

    def phi_large(n):
        if n <= LIMIT:
            return int(phi[n])
        facs = factor(n)
        result = n
        for p in facs:
            result -= result // p
        return result

    def get_phi(n):
        if n <= LIMIT:
            return int(phi[n])
        else:
            return phi_large(n)

    best_k = 1
    best_s = 1

    # 4. Dense scan of small range using sliding window
    max_scan = min(LIMIT - 200, 500000)
    for s in range(1, max_scan + 1):
        if not time_ok():
            break
        seen = set()
        k = 0
        while k < 200:
            v = int(phi[s + k])
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s

    # 5. Smart search for larger s
    max_n = 10**12
    random.seed(42)

    def check_window(s):
        seen = set()
        k = 0
        while k < 200 and s + k <= max_n:
            v = get_phi(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Generate candidates: numbers that are products of several small primes
    def generate_candidate(n_max):
        """Generate a candidate number with multiple prime factors."""
        num_primes = random.randint(2, 5)
        available_primes = primes[:100]
        chosen = random.sample(available_primes, min(num_primes, len(available_primes)))
        result = 1
        for p in chosen:
            power = random.randint(1, 3)
            result *= p ** power
        if result < 10**6:
            result *= random.randint(1, max(1, 10**12 // result))
        elif result > n_max:
            return random.randint(10**6, n_max)
        return min(result, n_max)

    # Local search: if we find a good position, check neighbors
    def local_search(s, depth=10):
        best_local_k = 0
        best_local_s = s
        for offset in range(-depth, depth + 1):
            test_s = s + offset
            if test_s < 1 or test_s + 200 > max_n:
                continue
            if not time_ok():
                break
            k = check_window(test_s)
            if k > best_local_k:
                best_local_k = k
                best_local_s = test_s
        return best_local_k, best_local_s

    # Phase 1: Random search with candidate generation
    attempts = 0
    while time_ok() and time.time() - start_time < DEADLINE - 10:
        attempts += 1
        if attempts % 3 == 0:
            s = generate_candidate(max_n - 200)
        else:
            s = random.randint(10**6, max_n - 200)

        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s
            if time.time() - start_time < DEADLINE - 5:
                local_k, local_s = local_search(s, depth=5)
                if local_k > best_k:
                    best_k = local_k
                    best_s = local_s

    # Phase 2: Focused search around the best position found
    if best_k > 1 and time.time() - start_time < DEADLINE - 5:
        local_k, local_s = local_search(best_s, depth=20)
        if local_k > best_k:
            best_k = local_k
            best_s = local_s

    return {'start': best_s, 'length': best_k}
