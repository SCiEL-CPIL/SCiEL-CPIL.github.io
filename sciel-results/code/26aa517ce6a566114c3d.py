"""Rewritten program: exact DP over subsets for minimum cochromatic partition."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Build edge set for O(1) lookup
    es = set()
    for a, b in P['edges']:
        es.add((a, b))
        es.add((b, a))

    # Precompute adjacency as bitmasks
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    full_mask = (1 << n) - 1

    # Precompute which subsets are cliques or independent sets
    # is_good[mask] = True if the induced subgraph on mask is a clique or independent set
    # We compute this incrementally using DP.
    # For a subset S, it is a clique if every vertex in S is adjacent to all others in S.
    # It is an independent set if no two vertices in S are adjacent.

    # Strategy: for each subset, pick the lowest set bit v, and check if S\{v} is good
    # and v is compatible (adjacent to all for clique, non-adjacent to all for indep).
    # But a subset might be a clique OR an independent set. We need to track which.

    # Let's use two boolean arrays:
    # is_clique[mask]: True if induced subgraph on mask is a clique
    # is_indep[mask]: True if induced subgraph on mask is an independent set

    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)

    # Base case: empty set and single vertices
    is_clique[0] = True
    is_indep[0] = True
    for v in range(n):
        m = 1 << v
        is_clique[m] = True
        is_indep[m] = True

    # Fill in for all masks from 1 to full_mask
    # For mask > 0, let v = lowest set bit, rest = mask ^ (1<<v)
    # is_clique[mask] = is_clique[rest] AND (v is adjacent to every vertex in rest)
    # is_indep[mask] = is_indep[rest] AND (v is NOT adjacent to any vertex in rest)

    for mask in range(1, 1 << n):
        # Skip if already computed (single vertices and 0)
        if mask & (mask - 1) == 0:  # single bit or zero
            continue
        v_bit = mask & (-mask)
        v = v_bit.bit_length() - 1
        rest = mask ^ v_bit

        # Check clique: v must be adjacent to all in rest
        # v is adjacent to all in rest iff (adj[v] & rest) == rest
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True

        # Check independent: v must NOT be adjacent to any in rest
        # v is non-adjacent to all in rest iff (adj[v] & rest) == 0
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True

    # Now is_good[mask] = is_clique[mask] or is_indep[mask]
    is_good = [False] * (1 << n)
    for mask in range(1 << n):
        is_good[mask] = is_clique[mask] or is_indep[mask]

    # DP: dp[mask] = minimum number of parts to cover the set of vertices in mask
    # dp[0] = 0
    # dp[mask] = min over all nonempty good submasks S of mask: 1 + dp[mask ^ S]
    # To make this efficient, we iterate over all good submasks.
    # For n=14, 2^14 = 16384 states. For each state, we iterate over submasks.
    # Total is O(3^n) which is about 4.78 million for n=14. This is very fast.

    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)  # parent[mask] = the good submask chosen

    for mask in range(1, 1 << n):
        # Enumerate all nonempty submasks of mask
        sub = mask
        while sub > 0:
            if is_good[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask

    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask > 0:
        sub = parent[mask]
        if sub < 0:
            # Fallback: should not happen
            parts.append([i for i in range(n) if (mask >> i) & 1])
            break
        # Convert submask to list of vertices
        part = [i for i in range(n) if (sub >> i) & 1]
        parts.append(part)
        mask ^= sub

    # Ensure types are correct: list of lists of Python ints
    # Convert any numpy ints to Python ints if needed
    result = []
    for part in parts:
        result.append([int(v) for v in part])

    return result

# EVOLVE-BLOCK-END
