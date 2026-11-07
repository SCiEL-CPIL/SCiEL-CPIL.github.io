import time
from math import isqrt, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 256

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # For each prime, find the largest prime power p^k <= N
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Baseline: 1 + all prime powers
    baseline = [1] + [prime_powers[p] for p in primes]
    baseline_sum = sum(baseline)

    # Identify all possible merge options:
    # A merge option is a set of primes whose product <= N,
    # and the product gives more sum than the individual prime powers.
    # We consider 2-way and 3-way merges.

    merge_options = []  # (gain, frozenset_of_primes, product_value)

    # 2-way merges
    for i in range(len(primes)):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            Pq = prime_powers[q]
            prod = p * q
            if prod > N:
                break
            gain = prod - Pp - Pq
            if gain > 0:
                merge_options.append((gain, frozenset([p, q]), prod))

    # 3-way merges
    for i in range(len(primes)):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            Pq = prime_powers[q]
            pq = p * q
            if pq > N:
                break
            for k in range(j + 1, len(primes)):
                r = primes[k]
                Pr = prime_powers[r]
                pqr = pq * r
                if pqr > N:
                    break
                gain = pqr - Pp - Pq - Pr
                if gain > 0:
                    merge_options.append((gain, frozenset([p, q, r]), pqr))

    # Sort by gain descending
    merge_options.sort(key=lambda x: x[0], reverse=True)

    # Backtracking to find optimal set of non-conflicting merges
    num_options = len(merge_options)
    best_gain = 0
    best_selected = []

    # Precompute suffix sums of gains for pruning
    suffix_gains = [0] * (num_options + 1)
    for i in range(num_options - 1, -1, -1):
        suffix_gains[i] = suffix_gains[i + 1] + merge_options[i][0]

    def backtrack(idx, used_primes, current_gain, selected):
        nonlocal best_gain, best_selected

        # Time check
        if time.time() > deadline:
            return

        # Pruning
        if current_gain + suffix_gains[idx] <= best_gain:
            return

        if idx == num_options:
            if current_gain > best_gain:
                best_gain = current_gain
                best_selected = list(selected)
            return

        # Option 1: Skip candidate idx
        backtrack(idx + 1, used_primes, current_gain, selected)

        # Option 2: Take candidate idx
        gain, pf, prod = merge_options[idx]
        # Check conflict
        if used_primes.isdisjoint(pf):
            new_used = used_primes | pf
            selected.append(idx)
            backtrack(idx + 1, new_used, current_gain + gain, selected)
            selected.pop()

    backtrack(0, set(), 0, [])

    # Construct Result
    selected_primes = set()
    result = [1]

    for idx in best_selected:
        gain, pf, prod = merge_options[idx]
        result.append(prod)
        selected_primes.update(pf)

    for p in primes:
        if p not in selected_primes:
            result.append(prime_powers[p])

    # Verification
    # Check distinct
    if len(set(result)) != len(result):
        result = baseline

    # Check range
    for x in result:
        if x < 1 or x > N:
            result = baseline
            break

    # Check coprime
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        result = baseline

    return [int(x) for x in result]
