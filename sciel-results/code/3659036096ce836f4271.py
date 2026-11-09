import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 150s limit

    best_M = 333312
    best_ks = [336, 372, 434]

    def linear_sigma_sieve(N):
        """Compute sigma(n) for all n up to N in O(N) time using linear sieve."""
        sigma = [0] * (N + 1)
        spf = [0] * (N + 1)
        prime_power = [0] * (N + 1)
        sigma_prime_power = [0] * (N + 1)
        primes = []

        sigma[1] = 1
        spf[1] = 1
        prime_power[1] = 1
        sigma_prime_power[1] = 1

        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
                prime_power[i] = i
                sigma_prime_power[i] = 1 + i
                sigma[i] = 1 + i
            for p in primes:
                if p > spf[i] or i * p > N:
                    break
                ip = i * p
                spf[ip] = p
                if p == spf[i]:
                    pp = prime_power[i] * p
                    prime_power[ip] = pp
                    ssp = sigma_prime_power[i] + pp
                    sigma_prime_power[ip] = ssp
                    sigma[ip] = sigma[i] // sigma_prime_power[i] * ssp
                    break
                else:
                    prime_power[ip] = p
                    sigma_prime_power[ip] = 1 + p
                    sigma[ip] = sigma[i] * (1 + p)

        return sigma

    def find_best_bruteforce(N):
        """Brute force: enumerate k up to N, find M with most k solutions."""
        sigma = linear_sigma_sieve(N)
        m_to_k = defaultdict(list)
        for k in range(1, N + 1):
            M_val = k * sigma[k]
            if M_val > 10**18:
                continue
            if k <= math.isqrt(M_val):
                m_to_k[M_val].append(k)

        best_M_local = 0
        best_ks_local = []
        for M_val, ks in m_to_k.items():
            if len(ks) > len(best_ks_local):
                best_ks_local = ks
                best_M_local = M_val
        return best_M_local, best_ks_local

    def get_divisors(n):
        """Get all divisors of n."""
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs

    def sigma_from_factorization(factors):
        """Compute sigma(n) from prime factorization {p: a}."""
        result = 1
        for p, a in factors.items():
            result *= (p ** (a + 1) - 1) // (p - 1)
        return result

    def factorize(n):
        """Factorize n using trial division + Miller-Rabin for large primes."""
        factors = {}
        temp = n
        # Trial division for small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            # Check if temp is prime
            if is_prime(temp):
                factors[temp] = factors.get(temp, 0) + 1
            else:
                # For composite, try more trial division
                p = 101
                while p * p <= temp:
                    while temp % p == 0:
                        factors[p] = factors.get(p, 0) + 1
                        temp //= p
                    p += 2
                if temp > 1:
                    factors[temp] = factors.get(temp, 0) + 1
        return factors

    def is_prime(n):
        """Miller-Rabin primality test."""
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Test with witnesses
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def find_solutions_for_M(M, max_k_check):
        """Find all k <= sqrt(M) such that k * sigma(k) = M."""
        if M < 1:
            return []
        sqrt_M = math.isqrt(M)
        if sqrt_M < 1:
            return []
        
        # Get divisors of M up to sqrt_M
        divs = []
        i = 1
        while i * i <= M:
            if M % i == 0:
                if i <= sqrt_M:
                    divs.append(i)
                other = M // i
                if other != i and other <= sqrt_M:
                    divs.append(other)
            i += 1
        
        solutions = []
        for k in divs:
            if k > max_k_check:
                continue
            # Check if sigma(k) == M // k
            target_sigma = M // k
            # Factorize k and compute sigma
            factors = factorize(k)
            s = sigma_from_factorization(factors)
            if s == target_sigma:
                solutions.append(k)
        
        return solutions

    def generate_candidate_M_values():
        """Generate candidate M values that are likely to have many solutions."""
        candidates = set()
        
        # Primorials and products of small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        # Products of first n primes (primorials)
        product = 1
        for i, p in enumerate(small_primes):
            if product > 10**18:
                break
            product *= p
            if product <= 10**18:
                candidates.add(product)
        
        # Products with various exponents of small primes
        # Focus on 2, 3, 5 with high exponents
        for a in range(0, 62):  # 2^61 ~ 2.3e18
            for b in range(0, 40):  # 3^39 ~ 4e18
                for c in range(0, 26):  # 5^25 ~ 2.9e17
                    M = (2 ** a) * (3 ** b) * (5 ** c)
                    if M > 10**18:
                        break
                    if M >= 1:
                        candidates.add(M)
                if (2 ** a) * (3 ** b) > 10**18:
                    break
            if 2 ** a > 10**18:
                break
        
        # Also try with 7
        for a in range(0, 60):
            for b in range(0, 38):
                for c in range(0, 24):
                    for d in range(0, 18):
                        M = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                        if M > 10**18:
                            break
                        if M >= 1:
                            candidates.add(M)
                    if (2 ** a) * (3 ** b) * (5 ** c) > 10**18:
                        break
                if (2 ** a) * (3 ** b) > 10**18:
                    break
            if 2 ** a > 10**18:
                break
        
        return candidates

    # Phase 1: Brute force with linear sieve
    for N in [200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000]:
        if time.time() - start_time > DEADLINE:
            break
        t0 = time.time()
        M, ks = find_best_bruteforce(N)
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
            if len(best_ks) >= 8:
                break
        if time.time() - t0 > 12:
            if N >= 5000000:
                break

    if time.time() - start_time > DEADLINE:
        return {'M': int(best_M), 'k': [int(k) for k in best_ks]}

    # Phase 2: Targeted search on candidate M values
    candidates = generate_candidate_M_values()
    # Sort by number of divisors (more divisors = more potential k values)
    # Approximate: prioritize M with many small prime factors
    candidate_list = sorted(candidates, key=lambda m: -len(get_divisors(m)) if m <= 10**12 else 0)
    
    # Limit candidates to avoid timeout
    max_candidates = 500
    checked = 0
    for M in candidate_list:
        if time.time() - start_time > DEADLINE:
            break
        if M > 10**18 or M < 1:
            continue
        if M > 10**14:  # Too large to enumerate divisors efficiently
            continue
        
        checked += 1
        if checked > max_candidates:
            break
        
        solutions = find_solutions_for_M(M, max_k_check=10**7)
        if len(solutions) > len(best_ks):
            best_M = M
            best_ks = solutions
            if len(best_ks) >= 10:
                break

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
