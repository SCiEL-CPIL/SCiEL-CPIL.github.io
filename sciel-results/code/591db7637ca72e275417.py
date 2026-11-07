"""Combined program: prime-power baseline with exact maximum-weight matching for merges,
including 2-way and 3-way composite merges."""
# EVOLVE-BLOCK-START
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

    # For each prime, its "cost" if not merged is its prime power value.
    # A merge of primes S gives value prod(S) and has gain = prod(S) - sum(prime_powers[p] for p in S)

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

    # Now we need to select a set of non-overlapping merge options to maximize total gain.
    # This is a weighted set packing problem. Since the number of primes involved in merges
    # is small (only primes where prime_powers are "small" enough to allow merges),
    # we can use branch and bound or DP.

    # Which primes can participate in merges?
    mergeable_primes = set()
    for gain, prime_set, prod in merge_options:
        mergeable_primes.update(prime_set)

    # Sort merge options by gain descending for better pruning
    merge_options.sort(key=lambda x: -x[0])

    # Use recursive backtracking with pruning to find optimal selection
    # The number of mergeable primes is small, so this is feasible.

    best_gain = [0]
    best_selection = [[]]  # list of (gain, prime_set, prod)

    # Map primes to indices for efficiency
    mergeable_list = sorted(mergeable_primes)
    prime_to_idx = {p: i for i, p in enumerate(mergeable_list)}
    num_mergeable = len(mergeable_list)

    # Represent selected primes as a bitmask
    # num_mergeable should be small (at most ~15 or so)

    if num_mergeable <= 20:
        # Bitmask DP / backtracking
        best = [0] * (1 << num_mergeable)
        # Actually, let's use simple backtracking with memoization on (option_idx, used_mask)

        # Better: just do DFS with pruning since options are few
        def dfs(idx, used_mask, current_gain, current_selection):
            if time.time() > deadline - 5:
                return
            if current_gain > best_gain[0]:
                best_gain[0] = current_gain
                best_selection[0] = list(current_selection)

            if idx >= len(merge_options):
                return

            # Pruning: even if we take all remaining options, can we beat best?
            # (optimistic upper bound)
            remaining_max = sum(opt[0] for opt in merge_options[idx:])
            if current_gain + remaining_max <= best_gain[0]:
                return

            # Option 1: skip this merge option
            dfs(idx + 1, used_mask, current_gain, current_selection)

            # Option 2: take this merge option if no conflict
            gain, prime_set, prod = merge_options[idx]
            # Check if any prime in prime_set is already used
            conflict = False
            new_mask = used_mask
            for p in prime_set:
                pi = prime_to_idx[p]
                bit = 1 << pi
                if used_mask & bit:
                    conflict = True
                    break
                new_mask |= bit
            if not conflict:
                current_selection.append(merge_options[idx])
                dfs(idx + 1, new_mask, current_gain + gain, current_selection)
                current_selection.pop()

        dfs(0, 0, 0, [])

    else:
        # Fallback to greedy if too many mergeable primes (shouldn't happen)
        used_primes = set()
        for gain, prime_set, prod in merge_options:
            if not any(p in used_primes for p in prime_set):
                best_selection[0].append((gain, prime_set, prod))
                best_gain[0] += gain
                used_primes.update(prime_set)

    # Construct final result from best selection
    selected_primes = set()
    for gain, prime_set, prod in best_selection[0]:
        selected_primes.update(prime_set)

    result = [1]
    for p in primes:
        if p not in selected_primes:
            result.append(prime_powers[p])

    for gain, prime_set, prod in best_selection[0]:
        result.append(prod)

    # Verify all constraints
    valid = True
    if len(set(result)) != len(result):
        valid = False
    for x in result:
        if not isinstance(x, int) or x < 1 or x > N:
            valid = False
            break
    if valid:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if gcd(result[i], result[j]) != 1:
                    valid = False
                    break
            if not valid:
                break

    if not valid:
        # Fallback to baseline
        result = baseline

    # Ensure we return Python ints in a list
    result = [int(x) for x in result]

    return result

# EVOLVE-BLOCK-END
