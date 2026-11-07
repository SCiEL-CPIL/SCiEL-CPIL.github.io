"""Optimized cochromatic partition via subset DP with bit-optimized clique/independent-set checks."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    t0 = time.time()
    deadline = t0 + 50  # safety margin before 60s limit

    # Build adjacency as bitmask per vertex
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    # For each vertex v, adj_mask[v] is the set of neighbors.
    # A set S (mask) is a clique iff for every v in S, S \ {v} is a clique AND S\{v} ⊆ adj_mask[v].
    # A set S is an independent set iff for every v in S, S \ {v} is an independent set AND S\{v} ∩ adj_mask[v] == 0.

    full_mask = (1 << n) - 1
    size = 1 << n

    # Precompute valid[mask] = True if mask is a clique or independent set
    is_valid = [False] * size
    is_clique = [False] * size
    is_ind = [False] * size

    # Singletons are both
    for v in range(n):
        is_clique[1 << v] = True
        is_ind[1 << v] = True
        is_valid[1 << v] = True

    # Process masks in increasing order of popcount (or just increasing mask value works since we use lsb)
    # For each mask, pick lsb vertex v, rest = mask ^ lsb
    # mask is clique iff is_clique[rest] AND (rest & ~adj_mask[v]) == 0  i.e., all of rest are neighbors of v
    # mask is ind iff is_ind[rest] AND (rest & adj_mask[v]) == 0  i.e., none of rest are neighbors of v

    for mask in range(2, size):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        # Check clique
        if is_clique[rest] and (rest & ~adj_mask[v] & full_mask) == 0:
            is_clique[mask] = True
            is_valid[mask] = True

        # Check independent set
        if is_ind[rest] and (rest & adj_mask[v]) == 0:
            is_ind[mask] = True
            is_valid[mask] = True

    # DP: dp[mask] = minimum number of valid subsets to partition mask
    INF = n + 1
    dp = [INF] * size
    dp[0] = 0
    parent = [0] * size

    for mask in range(1, size):
        if time.time() > deadline:
            break
        best = INF
        best_sub = 0
        # Iterate over all submasks of mask
        sub = mask
        while sub:
            if is_valid[sub]:
                val = dp[mask ^ sub] + 1
                if val < best:
                    best = val
                    best_sub = sub
            sub = (sub - 1) & mask

        dp[mask] = best
        parent[mask] = best_sub

    # Reconstruct partition
    partition = []
    mask = full_mask
    while mask:
        if time.time() > deadline:
            break
        sub = parent[mask]
        if sub == 0:
            # Fallback: add remaining vertices as singletons
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                partition.append([int(v)])
                mask ^= lsb
            break
        verts = []
        m = sub
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(int(v))
            m ^= lsb
        partition.append(verts)
        mask ^= sub

    # Verify all vertices covered exactly once
    covered = [0] * n
    for part in partition:
        for v in part:
            covered[int(v)] += 1

    if any(c != 1 for c in covered) or len(partition) == 0:
        # Greedy fallback
        es = set()
        for u, v in P['edges']:
            es.add((u, v))
            es.add((v, u))
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                k = len(trial)
                all_edges = True
                any_edge = False
                for i in range(k):
                    for j in range(i + 1, k):
                        if (trial[i], trial[j]) in es:
                            any_edge = True
                        else:
                            all_edges = False
                if all_edges or not any_edge:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    # Ensure all elements are Python int and lists are lists
    result = []
    for part in partition:
        result.append([int(v) for v in part])
    return result

# EVOLVE-BLOCK-END
