import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    full_mask = (1 << n) - 1

    # Bitmask adjacency
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute valid subsets (clique or independent set) using bitmask ops
    valid_set = set()
    for mask in range(1, 1 << n):
        m = mask
        clique = True
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if (adj[i] & mask) != (mask ^ lsb):
                clique = False
                break
            m ^= lsb
        if clique:
            valid_set.add(mask)
            continue
        m = mask
        indep = True
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if (adj[i] & mask) != 0:
                indep = False
                break
            m ^= lsb
        if indep:
            valid_set.add(mask)

    # DP: dp[mask] = min number of valid subsets to partition mask
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    choice = [0] * (1 << n)

    deadline = time.time() + 85.0
    completed = True

    for mask in range(1, 1 << n):
        if time.time() > deadline:
            completed = False
            break
        best = INF
        best_sub = 0
        sub = mask
        while sub:
            if sub in valid_set:
                prev = dp[mask ^ sub]
                if prev + 1 < best:
                    best = prev + 1
                    best_sub = sub
            if sub == 0:
                break
            sub = (sub - 1) & mask
        dp[mask] = best
        choice[mask] = best_sub

    if completed:
        parts = []
        mask = full_mask
        while mask:
            sub = choice[mask]
            if sub == 0:
                break
            verts = []
            m = sub
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            parts.append(verts)
            mask ^= sub
        return parts
    else:
        # Fallback greedy
        parts_masks = []
        used = 0
        verts = list(range(n))
        verts.sort(key=lambda x: adj[x].bit_count(), reverse=True)
        for v in verts:
            if used & (1 << v):
                continue
            found = False
            for i, p_mask in enumerate(parts_masks):
                new_mask = p_mask | (1 << v)
                m = new_mask
                clique = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if (adj[idx] & new_mask) != (new_mask ^ lsb):
                        clique = False
                        break
                    m ^= lsb
                if clique:
                    parts_masks[i] = new_mask
                    used |= (1 << v)
                    found = True
                    break
                m = new_mask
                indep = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if (adj[idx] & new_mask) != 0:
                        indep = False
                        break
                    m ^= lsb
                if indep:
                    parts_masks[i] = new_mask
                    used |= (1 << v)
                    found = True
                    break
            if not found:
                parts_masks.append(1 << v)
                used |= (1 << v)
        result = []
        for p_mask in parts_masks:
            verts = []
            m = p_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            result.append(verts)
        return result

# EVOLVE-BLOCK-END
