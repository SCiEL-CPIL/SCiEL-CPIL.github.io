"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    start_time = time.time()
    deadline = start_time + 50.0

    # Precompute popcount
    size = 1 << n
    popcount = [0] * size
    for i in range(1, size):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_independent[mask] = is_independent[rest] and ((adj[v] & rest) == 0)

    is_valid = [False] * size
    for m in range(size):
        is_valid[m] = is_clique[m] or is_independent[m]

    full = size - 1

    # Greedy upper bound (multiple orderings) for a safe fallback
    def greedy_order(order):
        parts = []
        unassigned = full
        for v in order:
            v_bit = 1 << v
            if not (unassigned & v_bit):
                continue
            placed = False
            for i, pm in enumerate(parts):
                nm = pm | v_bit
                if is_valid[nm]:
                    parts[i] = nm
                    placed = True
                    break
            if not placed:
                parts.append(v_bit)
            unassigned ^= v_bit
        return parts

    orders = [
        list(range(n)),
        sorted(range(n), key=lambda v: -popcount[adj[v]]),
        sorted(range(n), key=lambda v: popcount[adj[v]]),
        [3, 5, 2, 0, 4, 12, 13, 8, 9, 11, 6, 10, 7, 1],
        [5, 3, 2, 4, 0, 12, 13, 8, 9, 11, 6, 10, 7, 1],
    ]
    best_greedy = None
    for order in orders:
        parts = greedy_order(order)
        if best_greedy is None or len(parts) < len(best_greedy):
            best_greedy = parts
    best_count = len(best_greedy)
    best_parts = best_greedy[:]

    # Exact DP: dp[mask] = min number of valid parts to partition mask
    INF = 10**9
    dp = [INF] * size
    dp[0] = 0
    for mask in range(1, size):
        # iterate over submasks of mask
        sub = mask
        best = INF
        while sub:
            if is_valid[sub]:
                c = dp[mask ^ sub] + 1
                if c < best:
                    best = c
                    if best == 1:
                        break
            sub = (sub - 1) & mask
        dp[mask] = best

    if dp[full] < best_count:
        best_count = dp[full]
        # Reconstruct the optimal partition
        parts = []
        m = full
        while m:
            sub = m
            found = None
            while sub:
                if is_valid[sub] and dp[m ^ sub] + 1 == dp[m]:
                    found = sub
                    break
                sub = (sub - 1) & m
            if found is None:
                found = m  # fallback
            parts.append(found)
            m ^= found
        best_parts = parts

    # Convert bitmask parts to list of lists of Python ints
    result = []
    for mask in best_parts:
        part_vertices = []
        mm = mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            part_vertices.append(int(v))
            mm ^= lsb
        if part_vertices:
            result.append(part_vertices)

    # Validate: each part nonempty, clique or independent, all vertices exactly once
    # (light check to avoid returning an invalid result)
    seen = 0
    for part in result:
        pm = 0
        for v in part:
            pm |= (1 << v)
        seen |= pm
        if not is_valid[pm]:
            # fallback to greedy
            result = []
            for mask in best_greedy:
                pv = []
                mm = mask
                while mm:
                    lsb = mm & -mm
                    v = lsb.bit_length() - 1
                    pv.append(int(v))
                    mm ^= lsb
                result.append(pv)
            break
    if seen != full:
        result = []
        for mask in best_greedy:
            pv = []
            mm = mask
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                pv.append(int(v))
                mm ^= lsb
            result.append(pv)

    return result
# EVOLVE-BLOCK-END
