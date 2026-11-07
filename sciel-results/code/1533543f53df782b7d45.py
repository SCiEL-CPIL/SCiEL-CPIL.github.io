import time
import math
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    max_m = 10**18

    # Linear sieve for sigma(k) up to N
    # sigma(k) = sum of divisors of k
    # For prime p: sigma(p) = p+1
    # For k = p^a * m where gcd(p,m)=1: sigma(k) = sigma(p^a) * sigma(m)
    # We use the standard linear sieve approach with:
    #   lp[n] = largest prime factor of n
    #   pexp[n] = p^a where p = lp[n] and a is the exponent of p in n
    #   sigma[n] = sigma(n)
    
    # Choose N based on available time. Start with a reasonable N.
    # sigma(k) >= k+1, so k*sigma(k) >= k*(k+1). For k*sigma(k) <= 10^18,
    # we need k*(k+1) <= 10^18, so k < ~10^9. But we'll use a smaller N
    # and supplement with factorization for finding all solutions.
    
    N = 5000000  # 5 million - should run in ~5-10 seconds
    
    def linear_sigma_sieve(N):
        """Compute sigma(k) for all k from 1 to N using linear sieve."""
        sigma = [0] * (N + 1)
        sigma[1] = 1
        lp = [0] * (N + 1)  # largest prime factor
        pexp = [0] * (N + 1)  # p^a for the largest prime factor
        primes = []
        
        for i in range(2, N + 1):
            if lp[i] == 0:
                lp[i] = i
                pexp[i] = i
                sigma[i] = i + 1
                primes.append(i)
            for p in primes:
                if p > lp[i] or i * p > N:
                    break
                lp[i * p] = p
                if p == lp[i]:
                    pexp[i * p] = pexp[i] * p
                    # sigma(i*p) = sigma(i) - sigma(i/p^a) + sigma(p^(a+1))
                    # where p^a is the highest power of p dividing i
                    # sigma(p^(a+1)) = sigma(p^a) * (p + 1) - ... 
                    # Actually: sigma(i*p) = sigma(i) * (1 + p) - sigma(i/p^a) * p^(a+1)
                    # Simpler: sigma(i*p) = sigma(i/p^a) * sigma(p^(a+1)) * sigma(i / (p^a))
                    # Let's use: if i = p^a * m, gcd(p,m)=1
                    # sigma(i) = sigma(p^a) * sigma(m)
                    # sigma(i*p) = sigma(p^(a+1)) * sigma(m)
                    # = sigma(p^a) * (1+p) * sigma(m) - ... no
                    # sigma(p^(a+1)) = 1 + p + ... + p^(a+1) = (p^(a+2)-1)/(p-1)
                    # sigma(p^a) = (p^(a+1)-1)/(p-1)
                    # sigma(p^(a+1)) = sigma(p^a) + p^(a+1)
                    # So sigma(i*p) = sigma(m) * (sigma(p^a) + p^(a+1))
                    # = sigma(i) + sigma(m) * p^(a+1)
                    # sigma(m) = sigma(i) / sigma(p^a)
                    # This is getting complex. Let me use a different approach.
                    # 
                    # Alternative: sigma(i*p) = sigma(i) + sigma(i) - sigma(i/p^a) * p^(a+1)
                    # No, let me just compute it directly.
                    # 
                    # Actually the cleanest: 
                    # If i = p^a * m with gcd(p,m)=1:
                    # sigma(i) = (1+p+...+p^a) * sigma(m)
                    # i*p = p^(a+1) * m
                    # sigma(i*p) = (1+p+...+p^(a+1)) * sigma(m)
                    # = (1+p+...+p^a + p^(a+1)) * sigma(m)
                    # = sigma(i) + p^(a+1) * sigma(m)
                    # sigma(m) = sigma(i) // (1+p+...+p^a) = sigma(i) // (pexp[i]... )
                    # Hmm, we need sigma(m). 
                    # 
                    # Let me store an additional array: rest_sigma[i] = sigma(m) where i = p^a * m, p=lp[i]
                    # Or just use: sigma(i*p) = sigma(i) + pexp[i]*p * (sigma[i] // ((pexp[i]*p-1)//(p-1)))
                    # This is error-prone. Let me use a simpler but slightly less efficient method.
                    
                    # Actually, let's just use: 
                    # sigma(i*p) where p divides i:
                    # Let i = p^a * m, gcd(p,m)=1
                    # sigma(i) = S_a * sigma(m) where S_a = 1+p+...+p^a
                    # sigma(i*p) = S_{a+1} * sigma(m) = (S_a + p^(a+1)) * sigma(m)
                    # = sigma(i) + p^(a+1) * sigma(m)
                    # sigma(m) = sigma(i) / S_a
                    # S_a = (p^(a+1) - 1) / (p - 1) = (pexp[i]*p - 1)/(p-1)
                    # p^(a+1) = pexp[i] * p
                    # sigma(m) = sigma[i] * (p-1) / (pexp[i]*p - 1)
                    
                    S_a = (pexp[i] * p - 1) // (p - 1)
                    sigma_m = sigma[i] // S_a
                    sigma[i * p] = sigma[i] + pexp[i] * p * sigma_m
                else:
                    pexp[i * p] = p
                    sigma[i * p] = sigma[i] * (p + 1)
        
        return sigma, lp, pexp, primes
    
    # Run the sieve
    sigma, lp, pexp, primes = linear_sigma_sieve(N)
    
    # Now compute M = k * sigma(k) for all k in [2, N] and count occurrences
    m_counts = defaultdict(int)
    m_examples = defaultdict(list)
    
    for k in range(2, N + 1):
        m_val = k * sigma[k]
        if m_val <= max_m:
            m_counts[m_val] += 1
            if len(m_examples[m_val]) < 50:
                m_examples[m_val].append(k)
    
    # Sort by count
    sorted_ms = sorted(m_counts.items(), key=lambda item: item[1], reverse=True)
    
    if not sorted_ms:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Helper functions for factorization-based verification
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
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

    def compute_sigma(n, prime_factors=None):
        """Compute sigma(n) given its prime factorization or by factorizing n."""
        if n == 1:
            return 1
        if prime_factors is None:
            prime_factors = factorize(n)
        c = Counter(prime_factors)
        result = 1
        for p, e in c.items():
            result *= (p ** (e + 1) - 1) // (p - 1)
        return result

    def find_all_k_for_M(M):
        """Find all k <= isqrt(M) such that k * sigma(k) = M."""
        if M < 2:
            return []
        factors = factorize(M)
        divs = generate_divisors_from_factors(factors)
        m_primes = sorted(set(factors))
        valid_ks = []
        sqrt_m = math.isqrt(M)

        for d in divs:
            if d > sqrt_m or d < 1:
                continue
            # Compute sigma(d)
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

    # First, check the top candidates from the sieve
    num_candidates = min(50, len(sorted_ms))

    for i in range(num_candidates):
        if time.time() > deadline - 10:
            break

        M, count = sorted_ms[i]

        try:
            valid_ks = find_all_k_for_M(M)
            if len(valid_ks) > len(best_k_list):
                best_k_list = valid_ks
                best_m = M
        except Exception:
            continue

    # If we didn't find anything better, use the sieve examples
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
