import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def time_check():
        return time.time() < deadline

    N = 256

    # Sieve primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, 257):
        if i * i > N:
            break
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Max prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    n = len(primes)

    # Build adjacency with positive gains
    adj = [[] for _ in range(n)]
    for i in range(n):
        p = primes[i]
        for j in range(i + 1, n):
            q = primes[j]
            prod = p * q
            if prod > N:
                break
            g = prod - prime_powers[p] - prime_powers[q]
            if g > 0:
                adj[i].append((j, g))
                adj[j].append((i, g))

    # Find nodes with positive-degree edges
    active_nodes = set()
    for i in range(n):
        if adj[i]:
            active_nodes.add(i)

    # Map active nodes to compact indices
    active_list = sorted(active_nodes)
    node_map = {node: idx for idx, node in enumerate(active_list)}
    m = len(active_list)

    # Build weight matrix for compact graph
    w = [[0] * m for _ in range(m)]
    for i in range(n):
        if i not in node_map:
            continue
        ci = node_map[i]
        for j, g in adj[i]:
            if j in node_map:
                cj = node_map[j]
                w[ci][cj] = g
                w[cj][ci] = g

    # Iterative DP over bitmask for maximum weight matching
    # dp[mask] = max total gain from matching within the set of nodes in mask
    size = 1 << m
    dp = [0] * size

    # Iterate over masks in increasing order
    for mask in range(1, size):
        # Find lowest set bit
        i = 0
        while i < m and not (mask & (1 << i)):
            i += 1
        if i >= m:
            continue

        # Option 1: leave i unmatched
        best = dp[mask ^ (1 << i)]

        # Option 2: match i with some j in remaining
        rem = mask ^ (1 << i)
        j = i + 1
        while j < m:
            if rem & (1 << j):
                if w[i][j] > 0:
                    val = w[i][j] + dp[rem ^ (1 << j)]
                    if val > best:
                        best = val
            j += 1

        dp[mask] = best

    total_gain = dp[size - 1]

    # Reconstruct matching iteratively
    selected_pairs = []
    mask = size - 1
    while mask > 0:
        i = 0
        while i < m and not (mask & (1 << i)):
            i += 1
        if i >= m:
            break

        rem = mask ^ (1 << i)
        best_val = dp[rem]
        best_j = -1

        j = i + 1
        while j < m:
            if rem & (1 << j):
                if w[i][j] > 0:
                    val = w[i][j] + dp[rem ^ (1 << j)]
                    if val > best_val:
                        best_val = val
                        best_j = j
            j += 1

        if best_j >= 0:
            selected_pairs.append((i, best_j))
            mask = rem ^ (1 << best_j)
        else:
            mask = rem

    # Map back to original prime indices
    matched_prime_indices = set()
    product_pairs = []
    for ci, cj in selected_pairs:
        orig_i = active_list[ci]
        orig_j = active_list[cj]
        matched_prime_indices.add(orig_i)
        matched_prime_indices.add(orig_j)
        product_pairs.append((orig_i, orig_j))

    # Build result
    result = [1]

    # Add unmatched primes as their max prime power
    for i, p in enumerate(primes):
        if i not in matched_prime_indices:
            result.append(prime_powers[p])

    # Add matched products
    for i, j in product_pairs:
        result.append(primes[i] * primes[j])

    # Validate
    if len(set(result)) != len(result):
        return [1] + [prime_powers[p] for p in primes]

    for x in result:
        if not isinstance(x, int) or x < 1 or x > N:
            return [1] + [prime_powers[p] for p in primes]

    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [prime_powers[p] for p in primes]

    return result
