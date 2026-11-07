"""Cochromatic partition via optimized bitmask DP."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Precompute: for each subset, is it a clique or independent set?
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)

    is_clique[0] = True
    is_indep[0] = True

    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True

        if is_indep[rest] and (rest & adj[v]) == 0:
            is_indep[mask] = True

    # A part is valid if it's a clique or independent set
    is_valid_part = [is_clique[m] or is_indep[m] for m in range(1 << n)]

    # DP: dp[mask] = minimum number of parts to cover vertices in mask
    INF = 100
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        best = INF
        # Enumerate submasks of mask that contain the lowest set bit
        lsb = mask & (-mask)
        sub = mask
        while sub > 0:
            if sub & lsb:  # only consider submasks containing the lowest bit
                if is_valid_part[sub]:
                    val = dp[mask ^ sub] + 1
                    if val < best:
                        best = val
                        if best == 1:
                            break
            sub = (sub - 1) & mask
        dp[mask] = best

    min_parts = dp[full_mask]

    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask > 0:
        lsb = mask & (-mask)
        sub = mask
        found = False
        while sub > 0:
            if sub & lsb and is_valid_part[sub] and dp[mask ^ sub] + 1 == dp[mask]:
                parts.append(sub)
                mask = mask ^ sub
                found = True
                break
            sub = (sub - 1) & mask
        if not found:
            # Fallback: single vertex
            parts.append(lsb)
            mask = mask ^ lsb

    # Convert bitmask parts to lists of vertices
    result = []
    for part_mask in parts:
        verts = []
        for v in range(n):
            if part_mask & (1 << v):
                verts.append(v)
        result.append(verts)

    # Validate the result
    def validate(parts):
        seen = [False] * n
        for part in parts:
            for v in part:
                if not isinstance(v, int) or isinstance(v, bool):
                    return False
                if seen[v]:
                    return False
                seen[v] = True
            # Check clique or independent
            is_c = True
            is_i = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    if adj[part[i]] & (1 << part[j]):
                        is_i = False
                    else:
                        is_c = False
            if not (is_c or is_i):
                return False
        return all(seen)

    if not validate(result):
        # Fallback: all singletons (always valid)
        result = [[v] for v in range(n)]

    return result
