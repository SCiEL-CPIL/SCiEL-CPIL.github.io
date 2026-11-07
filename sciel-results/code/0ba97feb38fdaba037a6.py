# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    full = (1 << n) - 1
    N = 1 << n

    # Build adjacency bitmask
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute is_clique and is_indep for all subsets
    is_clique = [False] * N
    is_indep = [False] * N
    is_clique[0] = True
    is_indep[0] = True
    for mask in range(1, N):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True

    # Build list of all valid non-empty subsets
    valid_sets = []
    valid_mask = [False] * N
    for m in range(1, N):
        if is_clique[m] or is_indep[m]:
            valid_sets.append(m)
            valid_mask[m] = True

    # DP: f[mask] = min number of valid parts to cover mask
    INF = 10**9
    f = [INF] * N
    f[0] = 0

    # For each mask, find best valid submask s of mask
    # Optimization: iterate masks in increasing order of popcount
    # and for each, check valid submasks
    for mask in range(1, N):
        if time.time() - t0 > DEADLINE:
            break
        best = INF
        # Iterate over valid submasks of mask
        s = mask
        while s:
            if valid_mask[s]:
                val = f[mask ^ s] + 1
                if val < best:
                    best = val
                    if best == 1:
                        break
            s = (s - 1) & mask
        f[mask] = best

    if f[full] == INF:
        # Fallback: greedy
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                part_clique = True
                part_indep = True
                for i in range(len(part)):
                    for j in range(i + 1, len(part)):
                        if (adj[part[i]] >> part[j]) & 1:
                            part_indep = False
                        else:
                            part_clique = False
                has_edge_any = False
                has_edge_all = True
                for u in part:
                    e = (adj[u] >> v) & 1
                    if e:
                        has_edge_any = True
                    else:
                        has_edge_all = False
                if part_clique and has_edge_all:
                    part.append(v)
                    placed = True
                    break
                if part_indep and not has_edge_any:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return [list(p) for p in parts]

    # Reconstruct
    parts = []
    mask = full
    while mask:
        s = mask
        chosen = None
        while s:
            if valid_mask[s] and f[mask ^ s] + 1 == f[mask]:
                chosen = s
                break
            s = (s - 1) & mask
        if chosen is None:
            chosen = mask
        verts = []
        m = chosen
        while m:
            lsb = m & (-m)
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        parts.append(verts)
        mask ^= chosen

    return [list(p) for p in parts]

# EVOLVE-BLOCK-END
