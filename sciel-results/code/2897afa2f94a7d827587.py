import time
import math
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin for 150s limit

    # --- Sieve primes up to 2e6 for trial division and segmented sieve ---
    SIEVE_LIMIT = 2_000_000
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(SIEVE_LIMIT + 1) if sieve[i]]
    small_prime_set = set(small_primes)

    # --- Deterministic Miller-Rabin for n < 2^64 ---
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # --- Pollard-Brent (for rare large-factor verification) ---
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            c = random.randint(1, n - 1)
            y = random.randint(0, n - 1)
            m = 128
            g, r, q = 1, 1, 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if 1 < g < n:
                return g

    def factor_full(n):
        """Full factorization of n, returning list of prime factors."""
        factors = []
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    # --- Core: segmented-sieve smoothness check for [base+1, base+k] ---
    # For a given prime p and base = p*p, we want to know for each offset 1..200
    # whether base+offset is p-smooth.
    #
    # Strategy:
    #   1. Trial-divide each number by all primes q <= min(p, SIEVE_LIMIT).
    #   2. After removing all small-prime factors, check the remainder:
    #      - If 1: smooth.
    #      - If prime <= p: smooth.
    #      - If composite: factor it; smooth iff all factors <= p.
    #
    # Key optimization: we process ALL 200 numbers together, doing trial
    # division by each small prime q across the whole block. This is a
    # segmented-sieve-like operation: for each q, we find which of the 200
    # numbers are divisible by q and divide them out.
    #
    # This avoids 200 separate trial-division loops and instead does
    # (number of small primes <= p) passes over 200 numbers.

    def check_run(p, base, max_k=200):
        """Return the maximum k such that base+1..base+k are all p-smooth."""
        if time.time() > deadline:
            return 0

        # rem[i] will hold the remaining part of (base + i) after trial division
        rem = [base + i for i in range(1, max_k + 1)]
        # Track which positions are still "alive" (not yet determined to be non-smooth)
        # Actually, we can't determine non-smoothness until we finish all trial division.
        # We just need to fully factor out all primes <= p, then check remainders.

        # Trial division by all primes q <= min(p, SIEVE_LIMIT)
        # For efficiency, iterate over small_primes and break when q > p
        for q in small_primes:
            if q > p:
                break
            # For each position i where rem[i] is divisible by q, divide out all factors of q
            # To speed up, only check positions where rem[i] >= q*q (if rem[i] < q*q and rem[i]%q==0,
            # then rem[i] == q, which we handle)
            for i in range(max_k):
                if rem[i] >= q:
                    while rem[i] % q == 0:
                        rem[i] //= q
                    if rem[i] == 1:
                        pass  # fully factored

        # Now check remainders
        k = 0
        for i in range(max_k):
            r = rem[i]
            if r == 1:
                k += 1
                continue
            # r is not 1; check if r is prime
            if is_prime_mr(r):
                if r <= p:
                    k += 1
                    continue
                else:
                    break
            # r is composite; factor it
            try:
                factors = factor_full(r)
                if all(f <= p for f in factors):
                    k += 1
                    continue
            except Exception:
                pass
            break
        return k

    # --- Heuristic scoring for prioritizing which p to check first ---
    # The idea: p-smooth numbers around p^2 have density ~ rho(2) ~ 0.306.
    # But there might be correlations or structural reasons why certain p
    # produce longer runs. One heuristic: check p where p^2 mod small primes
    # produces favorable residues (making p^2+i more likely to be smooth).
    #
    # A simpler and more effective heuristic: just check p in an order that
    # balances coverage. We'll use a priority queue with a random-ish score
    # that slightly favors medium-sized primes (where factorization is fast
    # enough but the numbers are large enough for interesting smoothness).

    # --- Search strategy:
    # Phase 1: Check all primes up to 50000 with the segmented-sieve method.
    #          These are fast because trial division is cheap.
    # Phase 2: Check a sample of primes in [50000, 10^6] (we have them from sieve).
    # Phase 3: If time remains, check random primes in [10^6, 10^9] using
    #          Miller-Rabin for primality and the segmented-sieve for smoothness.
    #
    # We use a best-first approach: maintain a priority queue of candidate
    # primes, scored by a heuristic, and process the highest-scoring first.

    best_k = 0
    best_p = 3

    # Phase 1: All primes up to 50000
    for p in small_primes:
        if p > 50000:
            break
        if time.time() > deadline:
            break
        base = p * p
        k = check_run(p, base)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Primes in [50000, 10^6] — sample them
    # We have small_primes up to 2*10^6. Let's check a subset.
    # For p in [50000, 10^6], trial division by primes up to p is expensive.
    # But the segmented sieve approach helps: we only need primes up to p,
    # and for each such prime we do at most 200 divisions.
    # Number of primes up to 10^6 is ~78000. 78000 * 200 = 15.6M operations.
    # That's too many for one p. But we can limit: for each p, we only iterate
    # over primes up to min(p, SIEVE_LIMIT) = p (since p <= 10^6 < 2*10^6).
    # Actually, the inner loop is over small_primes up to p, and for each we do
    # 200 checks. So it's pi(p) * 200 operations per p.
    # For p = 10^6, pi(p) ~ 78000, so 15.6M ops. In Python, ~1-2 seconds.
    # So we can check maybe 50-100 such primes in the remaining time.

    # Let's be strategic: check primes at the top end first (larger p might
    # have slightly different smoothness properties), and also some in the middle.
    # Actually, let's just check them in order but with a time guard.

    mid_primes = [p for p in small_primes if 50000 <= p <= 1_000_000]
    # Shuffle to add some randomness, but keep a deterministic seed for reproducibility
    # Actually, let's check them in descending order (larger p first) as a heuristic
    # that larger p might allow longer runs (more room for smoothness).
    for p in reversed(mid_primes):
        if time.time() > deadline - 10.0:
            break
        base = p * p
        k = check_run(p, base)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 3: Random primes in [10^6, 10^9]
    # Generate random candidates and test primality with MR.
    if time.time() < deadline - 15.0:
        rng = random.Random(42)
        tried = set()
        while time.time() < deadline - 15.0:
            # Pick a random odd number in [10^6, 10^9]
            candidate = rng.randrange(10**6, 10**9, 2) | 1
            if candidate in tried:
                continue
            tried.add(candidate)

            # Quick primality check
            if not is_prime_mr(candidate):
                continue

            p = candidate
            base = p * p
            k = check_run(p, base)
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
