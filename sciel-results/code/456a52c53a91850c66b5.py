import time
from math import isqrt, gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def time_check():
        return time.time() < deadline

    N = 256

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Calculate max prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    n = len(primes)

    # Find all pairs (p, q) with positive gain: p*q - M_p - M_q > 0
    valid_edges = []
    for i in range(n):
        p = primes[i]
        for j in range(i + 1, n):
            q = primes[j]
            prod = p * q
            if prod > N:
                break
            g = prod - prime_powers[p] - prime_powers[q]
            if g > 0:
                valid_edges.append((i, j, g))

    nodes_in_edges = set()
    for i, j, g in valid_edges:
        nodes_in_edges.add(i)
        nodes_in_edges.add(j)

    # Base result: 1 + all prime powers
    base_result = [1] + [prime_powers[p] for p in primes]

    if not nodes_in_edges:
        return base_result

    # Compact the subgraph
    subgraph_nodes = sorted(list(nodes_in_edges))
    node_to_idx = {node: i for i, node in enumerate(subgraph_nodes)}
    m = len(subgraph_nodes)

    # Weight matrix for the subgraph
    w = [[0] * m for _ in range(m)]
    for i, j, g in valid_edges:
        ui = node_to_idx[i]
        uj = node_to_idx[j]
        w[ui][uj] = g
        w[uj][ui] = g

    full_mask = (1 << m) - 1

    @lru_cache(maxsize=None)
    def dp(mask):
        """Max gain from matching on the subgraph induced by nodes in mask."""
        if mask == 0:
            return 0
        # Find lowest set bit
        i = (mask & -mask).bit_length() - 1
        # Option 1: leave i unmatched
        best = dp(mask ^ (1 << i))
        # Option 2: match i with some j in mask
        rem = mask ^ (1 << i)
        j = i + 1
        while j < m:
            if rem & (1 << j):
                if w[i][j] > 0:
                    val = w[i][j] + dp(rem ^ (1 << j))
                    if val > best:
                        best = val
            j += 1
        return best

    @lru_cache(maxsize=None)
    def dp_reconstruct(mask):
        """Returns (gain, frozenset_of_pairs) for optimal matching on mask."""
        if mask == 0:
            return 0, frozenset()
        i = (mask & -mask).bit_length() - 1
        rem = mask ^ (1 << i)
        # Option 1: unmatched
        val1, pairs1 = dp_reconstruct(rem)
        best_val = val1
        best_pairs = pairs1
        # Option 2: matched with j
        j = i + 1
        while j < m:
            if rem & (1 << j):
                if w[i][j] > 0:
                    val2, pairs2 = dp_reconstruct(rem ^ (1 << j))
                    total = w[i][j] + val2
                    if total > best_val:
                        best_val = total
                        best_pairs = pairs2 | frozenset([(i, j)])
            j += 1
        return best_val, best_pairs

    if not time_check():
        return base_result

    _, selected_pairs_sub = dp_reconstruct(full_mask)

    # Map back to original prime indices
    selected_prime_pairs = []
    for i_sub, j_sub in selected_pairs_sub:
        i_orig = subgraph_nodes[i_sub]
        j_orig = subgraph_nodes[j_sub]
        selected_prime_pairs.append((i_orig, j_orig))

    # Build result
    result = [1]
    matched_indices = set()
    for i, j in selected_prime_pairs:
        matched_indices.add(i)
        matched_indices.add(j)

    for i, p in enumerate(primes):
        if i not in matched_indices:
            result.append(prime_powers[p])

    for i, j in selected_prime_pairs:
        result.append(primes[i] * primes[j])

    # Validate
    if len(set(result)) != len(result):
        return base_result
    for x in result:
        if x < 1 or x > N:
            return base_result
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return base_result

    return result
