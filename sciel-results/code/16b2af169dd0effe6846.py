import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    best_N = 2
    best_pairs = [[1, 1]]
    best_score = 1

    # Structural approach: N = R * s^2 where R is powerful.
    # For each Pythagorean pair (u, v) with u^2 + v^2 = s^2,
    # a = R*u^2, b = R*v^2 are both powerful (R powerful * square = powerful).
    # Number of such pairs is proportional to product of (2*e_p+1) for p|s, p≡1 mod 4.

    # Step 1: Find primes ≡ 1 mod 4 up to some limit
    def primes_1mod4(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [p for p in range(2, limit + 1) if sieve[p] and p % 4 == 1]

    # Primes ≡ 1 mod 4 up to 5000 (plenty for our purposes)
    p41 = primes_1mod4(5000)

    # Step 2: Search for optimal s = product of p_i^{e_i} maximizing (2*e_i+1) product
    # subject to s^2 * R <= 10^18, i.e., s <= 10^9 / sqrt(R)
    # We'll try R=1 first (simplest), then small powerful R values.

    def find_best_s(max_s):
        """Find s <= max_s maximizing the number of Pythagorean representations of s^2.
        Score = product of (2*e_p+1) for p|s with p≡1 mod 4.
        """
        best_val = 1
        best_s = 1
        best_factors = {}

        # DFS over primes ≡ 1 mod 4
        # Greedy: use small primes with higher exponents
        # For prime p with max exponent e_max, contribution is (2e+1) for cost p^e
        # We want to maximize product of (2e_i+1) subject to product p_i^{e_i} <= max_s

        # Use recursive search
        def dfs(idx, current_s, current_score, factors):
            nonlocal best_val, best_s, best_factors
            if current_score > best_val:
                best_val = current_score
                best_s = current_s
                best_factors = dict(factors)

            if idx >= len(p41):
                return

            p = p41[idx]
            # Try exponents 0, 1, 2, ...
            val = current_s
            exp = 0
            while val <= max_s:
                factors[p] = exp
                next_score = current_score * (2 * exp + 1)
                dfs(idx + 1, val, next_score, factors)
                exp += 1
                if exp == 0:
                    val = val * p
                else:
                    val = val * p
                if val > max_s:
                    break
            # Clean up
            if p in factors and factors[p] == 0:
                del factors[p]

        # Limit depth to keep search manageable
        # Only use first ~15 primes
        search_primes = p41[:20]
        # Override p41 temporarily
        global_p41 = p41
        p41_ref = p41
        # Use a local copy
        local_p41 = search_primes

        # Redefine dfs with local primes
        def dfs2(idx, current_s, current_score, factors):
            nonlocal best_val, best_s, best_factors
            if current_score > best_val:
                best_val = current_score
                best_s = current_s
                best_factors = dict(factors)

            if idx >= len(local_p41):
                return

            p = local_p41[idx]
            val = current_s
            exp = 0
            while val <= max_s:
                factors[p] = exp
                next_score = current_score * (2 * exp + 1)
                dfs2(idx + 1, val, next_score, factors)
                exp += 1
                val = val * p
                if val > max_s:
                    break
            if p in factors and factors[p] == 0:
                del factors[p]

        dfs2(0, 1, 1, {})
        return best_s, best_val, best_factors

    # Try R=1: s <= 10^9
    if time.time() < deadline - 30:
        s1, score1, factors1 = find_best_s(10**9)
        N1 = s1 * s1
        # Find Pythagorean pairs for s1
        pairs = []
        s1_sq = s1 * s1
        # u^2 + v^2 = s1^2, u <= v, u >= 1
        # u from 1 to s1//sqrt(2)
        max_u = int(math.isqrt(s1_sq // 2))
        for u in range(1, max_u + 1):
            v_sq = s1_sq - u * u
            v = int(math.isqrt(v_sq))
            if v * v == v_sq and v >= u:
                pairs.append([u * u, v * v])  # R=1, so a=u^2, b=v^2
            if time.time() > deadline - 20:
                break

        if len(pairs) > best_score:
            best_score = len(pairs)
            best_N = N1
            best_pairs = pairs

    # Try R=4 (powerful: 2^2): s <= 10^9/2 = 5*10^8
    if time.time() < deadline - 20:
        s2, score2, factors2 = find_best_s(5 * 10**8)
        N2 = 4 * s2 * s2
        pairs = []
        s2_sq = s2 * s2
        max_u = int(math.isqrt(s2_sq // 2))
        for u in range(1, max_u + 1):
            v_sq = s2_sq - u * u
            v = int(math.isqrt(v_sq))
            if v * v == v_sq and v >= u:
                pairs.append([4 * u * u, 4 * v * v])
            if time.time() > deadline - 10:
                break

        if len(pairs) > best_score:
            best_score = len(pairs)
            best_N = N2
            best_pairs = pairs

    # Try R=9 (powerful: 3^2): s <= 10^9/3
    if time.time() < deadline - 15:
        s3, score3, factors3 = find_best_s(10**9 // 3)
        N3 = 9 * s3 * s3
        pairs = []
        s3_sq = s3 * s3
        max_u = int(math.isqrt(s3_sq // 2))
        for u in range(1, max_u + 1):
            v_sq = s3_sq - u * u
            v = int(math.isqrt(v_sq))
            if v * v == v_sq and v >= u:
                pairs.append([9 * u * u, 9 * v * v])
            if time.time() > deadline - 5:
                break

        if len(pairs) > best_score:
            best_score = len(pairs)
            best_N = N3
            best_pairs = pairs

    # Try R=25 (5^2): s <= 10^9/5
    if time.time() < deadline - 10:
        s4, score4, factors4 = find_best_s(10**9 // 5)
        N4 = 25 * s4 * s4
        pairs = []
        s4_sq = s4 * s4
        max_u = int(math.isqrt(s4_sq // 2))
        for u in range(1, max_u + 1):
            v_sq = s4_sq - u * u
            v = int(math.isqrt(v_sq))
            if v * v == v_sq and v >= u:
                pairs.append([25 * u * u, 25 * v * v])
            if time.time() > deadline - 5:
                break

        if len(pairs) > best_score:
            best_score = len(pairs)
            best_N = N4
            best_pairs = pairs

    # Verify: check that all pairs are valid (a+b=N, a<=b, a,b powerful)
    # For safety, verify a few
    if best_pairs and len(best_pairs) > 0:
        # Verify first and last pair
        for idx in [0, len(best_pairs) - 1]:
            a, b = best_pairs[idx]
            assert a + b == best_N, f"Sum mismatch: {a}+{b} != {best_N}"
            assert a <= b, f"Order mismatch: {a} > {b}"
            # Verify powerful
            def is_powerful(n):
                if n == 1:
                    return True
                d = 2
                while d * d <= n:
                    if n % d == 0:
                        cnt = 0
                        while n % d == 0:
                            n //= d
                            cnt += 1
                        if cnt == 1:
                            return False
                    d += 1 if d == 2 else 2
                return True
            if not is_powerful(a) or not is_powerful(b):
                # Fallback
                return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': best_pairs}
