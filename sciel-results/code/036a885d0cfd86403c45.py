import time
import numpy as np
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    MAX_N = 256

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Precompute prime factors for each number
    prime_factors = {}
    for n in range(1, MAX_N + 1):
        factors = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        prime_factors[n] = factors

    # Precompute compatibility using bitmasks
    # compat_mask[n] is a bitmask where bit k is set if n and (k+1) are coprime
    compat_mask = [0] * (MAX_N + 1)
    for a in range(1, MAX_N + 1):
        mask = 0
        for b in range(1, MAX_N + 1):
            if a == b or gcd(a, b) == 1:
                mask |= (1 << (b - 1))
        compat_mask[a] = mask

    # Also precompute a set-based compat for quick lookup
    compat = [[False] * (MAX_N + 1) for _ in range(MAX_N + 1)]
    for a in range(1, MAX_N + 1):
        for b in range(1, MAX_N + 1):
            if a == b or gcd(a, b) == 1:
                compat[a][b] = True

    def is_compatible_with_all(n, S):
        """Check if n is coprime to all elements in S using bitmask."""
        if not S:
            return True
        s_mask = 0
        for x in S:
            s_mask |= (1 << (x - 1))
        return (compat_mask[n] & s_mask) == s_mask

    def is_valid_set(S):
        """Verify all pairs in S are coprime."""
        lst = list(S)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if not compat[lst[i]][lst[j]]:
                    return False
        return True

    # 1. Prime-power greedy: for each prime, pick largest prime power <= 256
    def prime_power_greedy():
        S = {1}
        for p in primes:
            # Find largest power of p <= 256
            best = p
            power = p
            while power * p <= MAX_N:
                power *= p
                best = power
            # Only add if compatible with current set
            if is_compatible_with_all(best, S):
                S.add(best)
        return S

    # 2. Descending greedy
    def greedy_descending():
        S = {1}
        for n in range(MAX_N, 1, -1):
            if is_compatible_with_all(n, S):
                S.add(n)
        return S

    # 3. Try ILP
    best_set = None
    best_sum = 0

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        A = np.zeros((num_primes, MAX_N), dtype=np.int8)
        for n in range(1, MAX_N + 1):
            for p in prime_factors[n]:
                A[prime_to_idx[p], n - 1] = 1

        c = -np.arange(1, MAX_N + 1, dtype=float)
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))
        integrality = np.ones(MAX_N)
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.status == 0:
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]
            if len(selected) > 0:
                if is_valid_set(set(selected)):
                    best_set = set(selected)
                    best_sum = sum(selected)
    except Exception:
        pass

    # 4. Try multiple greedy initializations
    candidates = [prime_power_greedy(), greedy_descending()]

    # Also try: start with primes, then try to upgrade to prime powers or products
    def prime_upgrade_greedy():
        S = set()
        S.add(1)
        # Start with all primes
        for p in primes:
            if p <= MAX_N:
                S.add(p)
        # Try to upgrade each prime to a larger compatible number
        for p in list(S):
            if p == 1:
                continue
            # Try to replace p with a larger number that's coprime to S\{p}
            for b in range(MAX_N, p, -1):
                if b in S:
                    continue
                ok = all(compat[b][x] for x in S if x != p)
                if ok:
                    S.discard(p)
                    S.add(b)
                    break
        return S

    candidates.append(prime_upgrade_greedy())

    for cand in candidates:
        cand_sum = sum(cand)
        if best_set is None or cand_sum > best_sum:
            best_set = cand
            best_sum = cand_sum

    # 5. Local search with bitmask-accelerated checks
    def local_search(S, S_sum, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            S_list = sorted(S, reverse=True)

            # Move type 1: Try adding a number if it fits
            for b in range(MAX_N, 1, -1):
                if b in S:
                    continue
                if is_compatible_with_all(b, S):
                    S.add(b)
                    S_sum += b
                    improved = True
                    break

            # Move type 2: Try 1-for-1 swaps: remove a, add b if b > a and compatible
            for a in list(S_list):
                if a == 1:
                    continue
                for b in range(MAX_N, a, -1):
                    if b in S:
                        continue
                    ok = all(compat[b][x] for x in S if x != a)
                    if ok:
                        new_sum = S_sum - a + b
                        if new_sum > S_sum:
                            S.discard(a)
                            S.add(b)
                            S_sum = new_sum
                            improved = True
                            break
                if improved:
                    break

            # Move type 3: Try 2-for-1 swaps: remove two small, add one large
            if len(S) >= 2:
                S_sorted = sorted(S)
                for i in range(len(S_sorted)):
                    a = S_sorted[i]
                    if a == 1:
                        continue
                    for j in range(i + 1, len(S_sorted)):
                        b = S_sorted[j]
                        if b == 1:
                            continue
                        if a + b >= MAX_N:
                            continue
                        for c in range(MAX_N, max(a, b), -1):
                            if c in S:
                                continue
                            ok = all(compat[c][x] for x in S if x != a and x != b)
                            if ok:
                                new_sum = S_sum - a - b + c
                                if new_sum > S_sum:
                                    S.discard(a)
                                    S.discard(b)
                                    S.add(c)
                                    S_sum = new_sum
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

            # Move type 4: Try 1-for-2 swaps: remove one, add two
            for a in list(S):
                if a == 1:
                    continue
                found = False
                for b in range(MAX_N, 1, -1):
                    if b in S or b == a:
                        continue
                    ok_b = all(compat[b][x] for x in S if x != a)
                    if not ok_b:
                        continue
                    for c in range(MAX_N, b, -1):
                        if c in S or c == a or c == b:
                            continue
                        if b + c <= a:
                            continue
                        ok_c = all(compat[c][x] for x in S if x != a)
                        if not ok_c or not compat[b][c]:
                            continue
                        new_sum = S_sum - a + b + c
                        if new_sum > S_sum:
                            S.discard(a)
                            S.add(b)
                            S.add(c)
                            S_sum = new_sum
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break

            # Move type 5: Try 2-for-2 swaps
            if len(S) >= 2 and time.time() < time_limit - 5:
                S_sorted = sorted(S)
                for i in range(len(S_sorted)):
                    a = S_sorted[i]
                    if a == 1:
                        continue
                    for j in range(i + 1, len(S_sorted)):
                        b = S_sorted[j]
                        if b == 1:
                            continue
                        # Try to find c, d not in S, c+d > a+b, both compatible with S\{a,b}
                        found = False
                        for c in range(MAX_N, max(a, b), -1):
                            if c in S:
                                continue
                            ok_c = all(compat[c][x] for x in S if x != a and x != b)
                            if not ok_c:
                                continue
                            for d in range(MAX_N, c, -1):
                                if d in S or d == c:
                                    continue
                                if c + d <= a + b:
                                    continue
                                ok_d = all(compat[d][x] for x in S if x != a and x != b)
                                if not ok_d or not compat[c][d]:
                                    continue
                                new_sum = S_sum - a - b + c + d
                                if new_sum > S_sum:
                                    S.discard(a)
                                    S.discard(b)
                                    S.add(c)
                                    S.add(d)
                                    S_sum = new_sum
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if improved:
                            break
                    if improved:
                        break

        return S, S_sum

    if best_set is not None:
        best_set, best_sum = local_search(best_set, best_sum, deadline - 3)

    # 6. Final polish
    if time.time() < deadline - 2:
        for b in range(MAX_N, 1, -1):
            if b in best_set:
                continue
            if is_compatible_with_all(b, best_set):
                best_set.add(b)
                best_sum += b

        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for a in list(best_set):
                if a == 1:
                    continue
                for b in range(MAX_N, a, -1):
                    if b in best_set:
                        continue
                    ok = all(compat[b][x] for x in best_set if x != a)
                    if ok:
                        new_sum = best_sum - a + b
                        if new_sum > best_sum:
                            best_set.discard(a)
                            best_set.add(b)
                            best_sum = new_sum
                            improved = True
                            break
                if improved:
                    break

    result = sorted(best_set)
    if not result:
        result = [1]

    return [int(x) for x in result]
