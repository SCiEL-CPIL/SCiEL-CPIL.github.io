import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    deadline = time.time() + 75  # safety margin before 90s kill

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute valid pieces (clique or independent set)
    # is_valid[mask] is True if the induced subgraph on mask is a clique or independent set
    is_valid = [False] * (1 << n)
    is_valid[0] = True

    for mask in range(1, 1 << n):
        is_clique = True
        is_indep = True

        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            rest = mask ^ lsb

            # Check clique condition for u: all others in rest must be neighbors of u
            if rest & ~adj[u]:
                is_clique = False

            # Check indep condition for u: no others in rest must be neighbors of u
            if rest & adj[u]:
                is_indep = False

            if not is_clique and not is_indep:
                break

            m ^= lsb

        is_valid[mask] = is_clique or is_indep

    # DP for minimum number of parts
    # dp[mask] = min parts to partition mask
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    best_sub = [0] * (1 << n)  # Store the submask used for reconstruction

    for mask in range(1, 1 << n):
        # Iterate submasks of mask
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    best_sub[mask] = sub
            sub = (sub - 1) & mask

    min_parts = dp[full_mask]

    # Reconstruct partition
    parts = []
    current_mask = full_mask
    while current_mask:
        sub = best_sub[current_mask]
        # Convert sub to list of vertices
        part_vertices = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        parts.append(part_vertices)
        current_mask ^= sub

    # Ensure types are correct: list of lists of Python ints
    result = []
    for part in parts:
        result.append([int(v) for v in part])

    return result
