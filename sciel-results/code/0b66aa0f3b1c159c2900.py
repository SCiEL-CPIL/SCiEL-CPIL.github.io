import math
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for p in MR_BASES:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in MR_BASES:
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = 2
            c = 3
            m = 128
            g = 1
            q = 1
            x = 0
            r = 0
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def factorize(n):
        """Return list of prime factors (with multiplicity) of n."""
        if n <= 1:
            return []
        factors = []
        # Trial division by small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
            else:
                d = pollard_brent(m)
                stack.append(d)
                stack.append(m // d)
        return factors

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        for f in factorize(n):
            if f > p:
                return False
        return True

    def check_window(p, start_k, max_k=200):
        """Check how many consecutive values p^2+1 .. p^2+max_k are p-smooth.
        Returns the count k."""
        k = 0
        for i in range(1, max_k + 1):
            if time.time() > deadline:
                break
            if not is_p_smooth(p * p + i, p):
                break
            k += 1
        return k

    # Strategy: search over primes p, prioritize larger p first since larger p
    # means more primes are allowed as factors, making p-smoothness easier.
    # But p^2 gets larger, so factorization is harder.
    # Let's try a mix: scan primes in descending order from a reasonable upper bound,
    # and also try some known good candidates.

    # First, let's try some heuristically good primes.
    # Known: for small p, the run is short. For larger p, runs can be longer.
    # Let's scan primes from high to low.

    best_k = 0
    best_p = 2

    # Generate primes up to some limit using simple sieve for primality testing
    # We'll test candidate primes and check their windows.

    # To be efficient, let's first check a set of candidate primes.
    # Start with primes in a range where p-smoothness is likely.

    # Try primes from 10^7 down to 10^5, then fill in.
    # Actually, let's be smarter: check primes in decreasing order of p,
    # but limit the search to keep within time budget.

    # Pre-compute a list of primes to test.
    # We'll generate primes using a simple method.

    # Let's test primes in this order:
    # 1. Large primes (e.g., 10^7 to 10^8) - fewer to test, each check is expensive
    # 2. Medium primes (10^5 to 10^7)
    # 3. Small primes (100 to 10^5)

    # For efficiency, let's generate primes up to 10^7 using sieve.
    limit = 10_000_000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))

    primes = [i for i in range(limit + 1) if sieve[i]]

    # Test primes in descending order (larger p first)
    # But we need to be careful: for very large p, factorizing p^2+i is expensive.
    # Let's cap p at something reasonable for the factorization to be fast.
    # With Pollard-Brent, factoring numbers up to ~10^16 should be feasible.
    # p^2 for p up to 10^8 is 10^16, which is at the edge.
    # Let's try p up to 10^7 first, then expand if time permits.

    # Test in descending order
    for p in reversed(primes):
        if time.time() > deadline:
            break
        if p < 2:
            break
        k = check_window(p, 1, 200)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    # If we haven't found a good one, also try some specific large primes
    # that might be good candidates.
    # Also try primes just above the sieve limit.
    for p in [10_000_019, 10_000_037, 10_000_081, 10_000_097, 10_000_103,
              10_000_121, 10_000_151, 10_000_179, 10_000_203, 10_000_209]:
        if time.time() > deadline:
            break
        if is_prime(p) and p > best_p:
            k = check_window(p, 1, 200)
            if k > best_k:
                best_k = k
                best_p = p

    # Try even larger primes
    for p in [100_000_007, 100_000_019, 100_000_079, 100_000_097, 100_000_127,
              200_000_003, 200_000_019, 300_000_001, 500_000_003, 1_000_000_007]:
        if time.time() > deadline:
            break
        if is_prime(p):
            k = check_window(p, 1, 200)
            if k > best_k:
                best_k = k
                best_p = p

    # Ensure we have a valid result
    if best_k < 1:
        # Fallback: find at least one prime where p^2+1 is p-smooth
        for p in primes:
            if time.time() > deadline:
                break
            if is_p_smooth(p * p + 1, p):
                best_k = 1
                best_p = p
                break

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
