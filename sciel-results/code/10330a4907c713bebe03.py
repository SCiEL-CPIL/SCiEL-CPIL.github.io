import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # Pre-compute smallest prime factor table for fast factorization
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def largest_prime_factor(n):
        if n <= 1:
            return 1
        max_pf = 1
        while n > 1:
            p = spf[n]
            if p > max_pf:
                max_pf = p
            while n % p == 0:
                n //= p
        return max_pf

    # Pre-compute P(n) for all n in [1, N]
    P = [0] * (N + 1)
    P[1] = 1
    for n in range(2, N + 1):
        P[n] = largest_prime_factor(n)

    # Phase 1: Smooth number search (fast, likely finds good solution)
    # Generate primes up to a reasonable limit
    def is_prime_simple(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = []
    candidate = 2
    while candidate <= 200:
        if is_prime_simple(candidate):
            primes.append(candidate)
        candidate += 1

    def generate_smooth(prime_list, limit):
        """Generate all numbers <= limit whose prime factors are all in prime_list."""
        smooths = {1}
        current = [1]
        for p in prime_list:
            new_vals = []
            for x in current:
                val = x
                while val * p <= limit:
                    val *= p
                    if val <= limit:
                        new_vals.append(val)
                        smooths.add(val)
                    else:
                        break
            current.extend(new_vals)
        return smooths

    best_pair = [1, N - 1]
    best_score = P[N - 1]  # P(1) = 1, so max is P(N-1)

    # Try increasing smoothness bounds
    for B_idx in range(1, len(primes)):
        if time.time() > deadline - 10:  # Leave 10s for fallback
            break
        prime_subset = primes[:B_idx + 1]
        smooths = generate_smooth(prime_subset, N)

        # Check pairs: find a in smooths such that N-a is also in smooths
        found = False
        for a in sorted(smooths):
            if a >= 1 and a <= N - 1:
                b = N - a
                if b in smooths:
                    score = max(P[a], P[b])
                    if score < best_score:
                        best_score = score
                        best_pair = [a, b]
                    found = True
                    break

        if found:
            # If we found a pair with small score, we can potentially stop
            # but let's try a few more to see if we can do better
            if best_score <= 10:
                break

    # Phase 2: Full scan if time permits
    if time.time() < deadline - 2:
        # Scan all a from 1 to N//2 to find optimal
        for a in range(1, N // 2 + 1):
            if time.time() > deadline - 2:
                break
            b = N - a
            score = max(P[a], P[b])
            if score < best_score:
                best_score = score
                best_pair = [a, b]
                if best_score == 2:
                    # Can't do better than 2 for odd N (both can't be powers of 2)
                    # Actually P(1)=1, so if a=1, score=P(N-1)
                    # Minimum possible is constrained by N being odd
                    break

    return best_pair
