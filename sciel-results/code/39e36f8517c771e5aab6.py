import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    best_a = 11
    best_b = 6
    best_error_num = 1
    best_error_den = 3

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

    def sigma_prime_power(p, e):
        """Compute sigma(p^e) = 1 + p + p^2 + ... + p^e"""
        total = 1
        power = 1
        for _ in range(e):
            power *= p
            total += power
        return total

    # Sieve to find primes up to a reasonable limit
    prime_limit = 10000
    sieve = [True] * (prime_limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(prime_limit)) + 1):
        if sieve[i]:
            for j in range(i * i, prime_limit + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, prime_limit + 1) if sieve[i]]

    # Strategy: For each prime p, consider a = p^a_exp * m, b = p^b_exp * m
    # where m is coprime to p. Then sigma(a) = sigma(p^a_exp) * sigma(m)
    # and sigma(b) = sigma(p^b_exp) * sigma(m).
    # So sigma(a) = sigma(b) iff sigma(p^a_exp) = sigma(p^b_exp).
    # This is only possible if a_exp = b_exp (since sigma(p^k) is strictly increasing in k for fixed p).
    # So this simple approach doesn't work directly.
    
    # Alternative: use two different primes.
    # a = p^a1 * q^a2, b = p^b1 * q^b2
    # sigma(a) = sigma(p^a1)*sigma(q^a2), sigma(b) = sigma(p^b1)*sigma(q^b2)
    # Need sigma(p^a1)*sigma(q^a2) = sigma(p^b1)*sigma(q^b2)
    # and a/b = (p^a1 * q^a2) / (p^b1 * q^b2) close to 3/2.
    
    # Even simpler: a = p * m, b = q * m where p, q are primes, m coprime to p*q.
    # sigma(a) = (1+p)*sigma(m), sigma(b) = (1+q)*sigma(m)
    # Equal iff 1+p = 1+q, i.e., p = q. Not useful.
    
    # Let's try: a = p^2 * m, b = q * m where p, q primes, m coprime.
    # sigma(a) = (1+p+p^2)*sigma(m), sigma(b) = (1+q)*sigma(m)
    # Need 1+p+p^2 = 1+q => q = p^2+p. Check if p^2+p is prime.
    # For p=2: q=6, not prime. p=3: q=12, not prime. p=5: q=30, not prime.
    # p^2+p = p(p+1) is always composite for p>=2. So this doesn't work.
    
    # Try: a = p * m, b = q^2 * m
    # sigma(a) = (1+p)*sigma(m), sigma(b) = (1+q+q^2)*sigma(m)
    # Need 1+p = 1+q+q^2 => p = q^2+q = q(q+1), composite. Doesn't work.
    
    # Let's use the direct search approach but more efficiently.
    # Compute sigma up to a limit using a simple sieve.
    
    def compute_sigma(L):
        """Compute sigma(n) for all n from 1 to L using sieve."""
        sigma = [0] * (L + 1)
        for i in range(1, L + 1):
            for j in range(i, L + 1, i):
                sigma[j] += i
        return sigma

    # Start with a moderate limit, then increase if time permits
    limits = [10000, 50000, 100000, 200000, 500000, 1000000, 2000000, 3000000, 5000000]

    for L in limits:
        if time.time() - start_time > deadline - 15:
            break

        try:
            sigma = compute_sigma(L)
        except MemoryError:
            break

        # Group by sigma value
        groups = {}
        for n in range(1, L + 1):
            s = sigma[n]
            if s not in groups:
                groups[s] = [n]
            else:
                groups[s].append(n)

        import bisect
        for s, nums in groups.items():
            if len(nums) < 2:
                continue
            if time.time() - start_time > deadline - 5:
                break

            nums_sorted = sorted(nums)
            n_elems = len(nums_sorted)

            for b in nums_sorted:
                target = 3 * b // 2
                # Find closest to target
                pos = bisect.bisect_left(nums_sorted, target)
                candidates = []
                if pos < n_elems:
                    candidates.append(nums_sorted[pos])
                if pos > 0:
                    candidates.append(nums_sorted[pos - 1])
                # Also check pos+1 and pos-2 for safety
                if pos + 1 < n_elems:
                    candidates.append(nums_sorted[pos + 1])
                if pos - 1 >= 0 and pos - 1 != pos - 1:  # already checked
                    pass

                for a in candidates:
                    if a == b:
                        continue
                    # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    # Compare num/den with best_error_num/best_error_den
                    if num * best_error_den < best_error_num * den:
                        best_error_num = num
                        best_error_den = den
                        best_a = a
                        best_b = b

        if time.time() - start_time > deadline - 10:
            break

    return [int(best_a), int(best_b)]
