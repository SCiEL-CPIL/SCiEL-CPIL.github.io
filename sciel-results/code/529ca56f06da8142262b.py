def construct():
    import math
    import time
    import heapq
    from collections import defaultdict

    start_time = time.time()
    deadline = start_time + 115.0

    # Primes up to ~100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    LIMIT = 10**18

    # Max-heap for top K candidates based on divisor count (tau)
    candidates = []
    K = 2000

    def get_factorization(n):
        """Return list of (prime, exponent) pairs."""
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        return factors

    def gen_divisors(factors):
        """Generate sorted divisors from factorization."""
        divs = [1]
        for p, exp in factors:
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        divs.sort()
        return divs

    def calc_score_fast(n):
        """Calculate tau_perp score efficiently."""
        if n < 2:
            return 0
        factors = get_factorization(n)
        divs = gen_divisors(factors)
        score = 0
        for j in range(len(divs) - 1):
            if math.gcd(divs[j], divs[j + 1]) == 1:
                score += 1
        return score

    def dfs(idx, n, div_count):
        if time.time() > deadline - 25.0:
            return

        if idx >= len(primes):
            if len(candidates) < K:
                heapq.heappush(candidates, (-div_count, n))
            else:
                if div_count > -candidates[0][0]:
                    heapq.heapreplace(candidates, (-div_count, n))
            return

        p = primes[idx]
        # Option 1: skip this prime
        dfs(idx + 1, n, div_count)

        # Option 2: include this prime with various exponents
        curr = n
        exp = 0
        while curr <= LIMIT:
            exp += 1
            # Max exponent heuristic: smaller primes can have higher exponents
            max_exp_allowed = max(1, int(60 / (idx + 1)))
            if exp > max_exp_allowed:
                break
            dfs(idx + 1, curr, div_count * (exp + 1))
            curr *= p
            if curr > LIMIT:
                break

    # Run DFS to find candidates
    dfs(0, 1, 1)

    # Extract top candidates
    top_candidates = []
    while candidates:
        neg_tau, n = heapq.heappop(candidates)
        top_candidates.append(n)

    best_n = 2
    best_score = 0

    # Evaluate primorial of first 15 primes as baseline
    primorial = 1
    for p in primes[:15]:
        primorial *= p
    if primorial <= LIMIT:
        s = calc_score_fast(primorial)
        if s > best_score:
            best_score = s
            best_n = primorial

    # Evaluate top candidates
    seen = set()
    seen.add(best_n)
    for n in top_candidates:
        if time.time() > deadline - 10.0:
            break
        if n in seen:
            continue
        seen.add(n)

        s = calc_score_fast(n)
        if s > best_score:
            best_score = s
            best_n = n

    # Local search: try modifying best_n
    def local_search(n, score, iterations=500):
        if time.time() > deadline - 5.0:
            return n, score

        factors = get_factorization(n)
        current_n = n
        current_score = score

        for _ in range(iterations):
            if time.time() > deadline - 5.0:
                break

            # Try removing a prime factor
            if len(factors) > 1:
                for i in range(len(factors)):
                    new_factors = factors[:i] + factors[i+1:]
                    new_n = 1
                    for p, e in new_factors:
                        new_n *= p ** e
                    if new_n <= LIMIT and new_n >= 2:
                        new_score = calc_score_fast(new_n)
                        if new_score > current_score:
                            current_n = new_n
                            current_score = new_score
                            factors = new_factors
                            break

            # Try adding a small prime not already in factors
            existing_primes = set(p for p, _ in factors)
            for p in primes:
                if p not in existing_primes:
                    new_n = current_n * p
                    if new_n <= LIMIT:
                        new_factors = factors + [(p, 1)]
                        new_score = calc_score_fast(new_n)
                        if new_score > current_score:
                            current_n = new_n
                            current_score = new_score
                            factors = new_factors
                            break

            # Try increasing exponent of a prime
            for i in range(len(factors)):
                p, e = factors[i]
                new_n = current_n * p
                if new_n <= LIMIT:
                    new_factors = factors.copy()
                    new_factors[i] = (p, e + 1)
                    new_score = calc_score_fast(new_n)
                    if new_score > current_score:
                        current_n = new_n
                        current_score = new_score
                        factors = new_factors
                        break

            # Try decreasing exponent of a prime
            for i in range(len(factors)):
                p, e = factors[i]
                if e > 1:
                    new_n = current_n // p
                    new_factors = factors.copy()
                    new_factors[i] = (p, e - 1)
                    new_score = calc_score_fast(new_n)
                    if new_score > current_score:
                        current_n = new_n
                        current_score = new_score
                        factors = new_factors
                        break

        return current_n, current_score

    # Apply local search to the best candidate
    best_n, best_score = local_search(best_n, best_score)

    # Also try local search on a few other top candidates
    for n in top_candidates[:20]:
        if time.time() > deadline - 5.0:
            break
        if n == best_n:
            continue
        s = calc_score_fast(n)
        if s >= best_score - 5:  # Only search if close to best
            ln, ls = local_search(n, s, iterations=200)
            if ls > best_score:
                best_n = ln
                best_score = ls

    return best_n
