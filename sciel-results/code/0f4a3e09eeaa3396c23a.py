"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency bitmask: adj_mask[v] has bits set for all neighbors of v
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    # Precompute valid_mask: bitmask -> True if the vertex set is a clique or independent set
    # A set S is a clique iff for every v in S, (adj_mask[v] & S) == (S ^ (1 << v))
    # A set S is independent iff for every v in S, (adj_mask[v] & S) == 0
    size = 1 << n
    valid = [False] * size
    valid[0] = True
    for mask in range(1, size):
        # Quick check: popcount <= 1 => always valid
        # Use bit tricks for clique/independent check
        is_clique = True
        is_indep = True
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            rest = mask ^ low
            # For clique: v must be adjacent to all in rest
            if (adj_mask[v] & rest) != rest:
                is_clique = False
                break
            # For independent: v must not be adjacent to any in rest
            if adj_mask[v] & rest:
                is_indep = False
                # Can't break early if we still need is_clique check... 
                # but if is_indep is False, we still need to check clique
                # Actually we can break if neither is possible
                if not is_clique:
                    break
            m ^= low
        if is_clique or is_indep:
            valid[mask] = True

    # DP: dp[mask] = min number of valid parts to partition 'mask'
    # choice[mask] = the valid submask chosen as one part
    dp = [0] * size
    choice = [0] * size
    dp[0] = 0

    for mask in range(1, size):
        best = n + 1
        best_sub = 0
        sub = mask
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                val = dp[prev] + 1
                if val < best:
                    best = val
                    best_sub = sub
                    if best == 1:
                        break  # can't do better than 1
            sub = (sub - 1) & mask
        dp[mask] = best
        choice[mask] = best_sub

    # Check deadline
    if time.time() > deadline:
        # Fallback: return a trivial valid partition (each vertex its own part)
        return [[i] for i in range(n)]

    # Reconstruct
    parts = []
    mask = (1 << n) - 1
    while mask:
        sub = choice[mask]
        vertices = []
        m = sub
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            vertices.append(v)
            m ^= low
        parts.append(vertices)
        mask ^= sub

    return parts
# EVOLVE-BLOCK-END
