# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s wall-clock kill
    n = P['n']
    full = (1 << n) - 1

    # adjacency bitmask
    adj = [0] * n
    es = P['edges']
    for a, b in es:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute for each subset: is it a clique? independent set?
    N = 1 << n
    is_clique = [False] * N
    is_indep = [False] * N
    is_clique[0] = True
    is_indep[0] = True
    for mask in range(1, N):
        # take lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        # clique: v connected to all in rest, and rest is clique
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
        # independent: v not connected to any in rest, and rest is independent
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True

    valid = [False] * N
    for m in range(N):
        if is_clique[m] or is_indep[m]:
            valid[m] = True

    # DP: f[mask] = min number of valid parts to partition mask
    INF = 10**9
    f = [INF] * N
    f[0] = 0
    # iterate masks; for each, iterate valid submasks
    for mask in range(1, N):
        best = INF
        # iterate submasks of mask
        s = mask
        while s:
            if valid[s]:
                val = f[mask ^ s] + 1
                if val < best:
                    best = val
            s = (s - 1) & mask
            if time.time() - t0 > DEADLINE:
                break
        f[mask] = best
        if time.time() - t0 > DEADLINE:
            break

    # If deadline hit before finishing, fall back to a greedy partition
    if f[full] == INF:
        # greedy fallback
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                # check if adding v keeps part clique or independent
                ok = True
                for u in part:
                    has_edge = (adj[u] >> v) & 1
                    # we need to know current type
                # recompute
                has_edge_any = False
                has_edge_all = True
                for u in part:
                    e = (adj[u] >> v) & 1
                    if e:
                        has_edge_any = True
                    else:
                        has_edge_all = False
                # also need internal consistency of part already valid
                # determine if part is clique or independent currently
                part_clique = True
                part_indep = True
                plist = list(part)
                for i in range(len(plist)):
                    for j in range(i+1, len(plist)):
                        e = (adj[plist[i]] >> plist[j]) & 1
                        if e:
                            part_indep = False
                        else:
                            part_clique = False
                if part_clique and has_edge_all:
                    part.append(v); placed = True; break
                if part_indep and not has_edge_any:
                    part.append(v); placed = True; break
            if not placed:
                parts.append([v])
        return parts

    # Reconstruct partition by backtracking
    parts = []
    mask = full
    while mask:
        # find a valid submask s such that f[mask] == f[mask^s]+1
        s = mask
        chosen = None
        while s:
            if valid[s] and f[mask ^ s] + 1 == f[mask]:
                chosen = s
                break
            s = (s - 1) & mask
        if chosen is None:
            # should not happen; fallback
            chosen = mask
        # convert chosen bitmask to list of vertices
        verts = []
        m = chosen
        while m:
            lsb = m & (-m)
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        parts.append(verts)
        mask ^= chosen

    # Return as list of lists of Python int
    return [list(p) for p in parts]

# EVOLVE-BLOCK-END
