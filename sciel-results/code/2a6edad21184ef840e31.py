import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    n = P['n']
    edges = P['edges']

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Precompute is_clique and is_ind for all subsets
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_valid = [False] * (1 << n)

    is_clique[0] = True
    is_ind[0] = True
    is_valid[0] = True

    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        else:
            is_clique[mask] = False

        if is_ind[rest] and (rest & adj[v]) == 0:
            is_ind[mask] = True
        else:
            is_ind[mask] = False

        is_valid[mask] = is_clique[mask] or is_ind[mask]

    # Precompute for each vertex v, the list of valid subsets containing v,
    # sorted by size descending
    valid_subs_for_v = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if is_valid[mask]:
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                valid_subs_for_v[v].append(mask)
                m ^= lsb

    for v in range(n):
        valid_subs_for_v[v].sort(key=lambda x: x.bit_count(), reverse=True)

    # Memoization
    memo = {}
    choice = {}

    def can_cover(rem, k):
        if time.time() > deadline:
            return False
        if rem == 0:
            return True
        if k == 0:
            return False
        if k == 1:
            if is_valid[rem]:
                choice[(rem, 1)] = rem
                return True
            else:
                return False

        key = (rem, k)
        if key in memo:
            return memo[key]

        # Pick vertex with fewest valid subsets that are subsets of rem
        best_v = -1
        best_count = float('inf')
        r = rem
        while r:
            lsb = r & -r
            v = lsb.bit_length() - 1
            cnt = 0
            for S in valid_subs_for_v[v]:
                if (S & rem) == S:
                    cnt += 1
                    if cnt >= best_count:
                        break
            if cnt < best_count:
                best_count = cnt
                best_v = v
                if cnt == 0:
                    break
            r ^= lsb

        if best_v == -1 or best_count == 0:
            memo[key] = False
            return False

        v = best_v
        lsb_v = 1 << v

        # Sort candidate subsets: prefer larger subsets (greedy heuristic)
        candidates = []
        for S in valid_subs_for_v[v]:
            if (S & rem) == S:
                if S & lsb_v:
                    candidates.append(S)
        
        # Sort by size descending, then by bitmask for determinism
        candidates.sort(key=lambda x: (-x.bit_count(), x))

        for S in candidates:
            if can_cover(rem ^ S, k - 1):
                memo[key] = True
                choice[key] = S
                return True

        memo[key] = False
        return False

    # Find minimum k
    min_k = 1
    found = False
    while min_k <= n:
        if time.time() > deadline:
            break
        memo.clear()
        choice.clear()
        if can_cover(full_mask, min_k):
            found = True
            break
        min_k += 1

    if not found:
        return [[i] for i in range(n)]

    # Reconstruct partition
    parts = []
    rem = full_mask
    k = min_k
    while rem:
        key = (rem, k)
        S = choice[key]
        part_vertices = []
        for i in range(n):
            if S & (1 << i):
                part_vertices.append(i)
        parts.append(part_vertices)
        rem ^= S
        k -= 1

    return parts
