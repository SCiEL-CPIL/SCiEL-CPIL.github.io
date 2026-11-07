import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s wall-clock kill
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

    # Quick check: is the full graph a clique or independent set?
    if valid[full]:
        vertices = list(range(n))
        return [vertices]

    # DP: f[mask] = min number of valid parts to partition mask
    INF = 10**9
    f = [INF] * N
    f[0] = 0

    for mask in range(1, N):
        if time.time() - t0 > DEADLINE:
            break
        best = INF
        # Iterate submasks of mask; check larger submasks first for early termination
        s = mask
        while s:
            if valid[s]:
                val = f[mask ^ s] + 1
                if val < best:
                    best = val
                    if best == 1:
                        break
            s = (s - 1) & mask
            if time.time() - t0 > DEADLINE:
                break
        f[mask] = best

    # If we finished the DP, we can reconstruct the solution
    if f[full] != INF:
        # Reconstruct the partition
        parts = []
        mask = full
        while mask:
            # Find a valid submask s such that f[mask] == f[mask^s] + 1
            s = mask
            while s:
                if valid[s] and f[mask ^ s] + 1 == f[mask]:
                    parts.append(s)
                    mask ^= s
                    break
                s = (s - 1) & mask
            if not s and not parts:
                # Should not happen if f[full] is correct
                break
            if time.time() - t0 > DEADLINE:
                break
        
        if len(parts) == f[full]:
            # Convert bitmasks to lists of vertices
            final_parts = []
            for p in parts:
                vertices = []
                for i in range(n):
                    if (p >> i) & 1:
                        vertices.append(i)
                final_parts.append(vertices)
            return final_parts

    # Fallback: Greedy partition
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            # Check if adding v keeps part clique or independent
            # Determine current type of part
            part_clique = True
            part_indep = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    u, w = part[i], part[j]
                    if (adj[u] >> w) & 1:
                        part_indep = False
                    else:
                        part_clique = False
            if part_clique:
                # Check if v is adjacent to all in part
                ok = True
                for u in part:
                    if not ((adj[v] >> u) & 1):
                        ok = False
                        break
                if ok:
                    part.append(v)
                    placed = True
                    break
            elif part_indep:
                # Check if v is non-adjacent to all in part
                ok = True
                for u in part:
                    if (adj[v] >> u) & 1:
                        ok = False
                        break
                if ok:
                    part.append(v)
                    placed = True
                    break
            else:
                # Part is neither clique nor independent? Should not happen if we maintain validity
                pass
        if not placed:
            parts.append([v])
    
    return parts
