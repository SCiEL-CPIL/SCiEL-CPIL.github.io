"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    TARGET = 123457

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes_all = sieve(2000)

    def largest_prime_factor(n):
        """Compute the largest prime factor of n via trial division."""
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp

    def generate_smooth_numbers(primes, limit):
        """
        Generate all numbers <= limit whose prime factors are all in primes.
        Returns a dict: number -> largest prime factor (among the primes used).
        Uses iterative multiplication approach.
        """
        if not primes:
            return {1: 1}
        
        smooth = {1: 1}
        for p in primes:
            new_smooth = {}
            # Copy existing entries
            for x, lpf in smooth.items():
                new_smooth[x] = lpf
            # Multiply existing by powers of p
            for x, lpf in list(smooth.items()):
                val = x
                while val * p <= limit:
                    val *= p
                    new_lpf = max(lpf, p)
                    if val not in new_smooth or new_smooth[val] < new_lpf:
                        new_smooth[val] = new_lpf
            smooth = new_smooth
        return smooth

    # Initial best: [1, TARGET-1]
    best_result = [1, TARGET - 1]
    best_score = largest_prime_factor(TARGET - 1)  # P(123456)

    # Try increasing smoothness bounds: use first k primes
    for k in range(1, len(primes_all) + 1):
        if time.time() > deadline - 5:
            break

        primes_subset = primes_all[:k]
        max_p = primes_subset[-1]

        smooth = generate_smooth_numbers(primes_subset, TARGET)
        
        if len(smooth) < 2:
            continue

        # Check all pairs
        found_at_this_level = False
        min_score_at_level = float('inf')
        best_pair_at_level = None

        for a, pa in smooth.items():
            if time.time() > deadline - 3:
                break
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth:
                pb = smooth[b]
                score = max(pa, pb)
                if score < min_score_at_level:
                    min_score_at_level = score
                    best_pair_at_level = (a, b)
                found_at_this_level = True

        if found_at_this_level:
            if min_score_at_level < best_score:
                best_score = min_score_at_level
                best_result = list(best_pair_at_level)
            break

    # Additional targeted search: try a = 2^i * 3^j * 5^k * 7^l for small exponents
    if time.time() < deadline - 3:
        def lpf(n):
            if n <= 1:
                return 1
            lp = 1
            d = 2
            while d * d <= n:
                while n % d == 0:
                    lp = d
                    n //= d
                d += 1
            if n > 1:
                lp = n
            return lp

        for i in range(0, 18):
            if time.time() > deadline - 3:
                break
            for j in range(0, 14):
                if time.time() > deadline - 3:
                    break
                a_val = (1 << i) * (3 ** j)
                if a_val >= TARGET:
                    break
                for k in range(0, 11):
                    a_val_k = a_val * (5 ** k)
                    if a_val_k >= TARGET:
                        break
                    for l in range(0, 9):
                        a_val_kl = a_val_k * (7 ** l)
                        if a_val_kl >= TARGET:
                            break
                        b = TARGET - a_val_kl
                        if b < 1:
                            break
                        pa = lpf(a_val_kl)
                        pb = lpf(b)
                        score = max(pa, pb)
                        if score < best_score:
                            best_score = score
                            best_result = [a_val_kl, b]

    return best_result

# EVOLVE-BLOCK-END
