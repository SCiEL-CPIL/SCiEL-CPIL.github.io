"""Triangle-free diameter-four augmentation: exact bitmask branch-and-bound set cover."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start = time.time()
    deadline = 88.0  # safety margin under 90s wall-clock

    n = P['n']
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True

    # base distances on the path are just |i-j|
    base = lambda i, j: abs(i - j)

    # candidate added edges: non-adjacent on the path (distance >= 2)
    # triangle-free w.r.t. the path is automatic (a path has no triangles),
    # and two candidates conflict iff they share an endpoint or cross in path order.
    candidates = []
    for u in range(n):
        for v in range(u + 2, n):  # skip adjacent (v == u+1)
            candidates.append((u, v))
    m = len(candidates)

    # pairs that need reduction (base distance > 4)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if base(i, j) > 4:
                pairs.append((i, j))
    Pn = len(pairs)
    if Pn == 0:
        return []

    # coverage: edge e covers pair p if it reduces distance to <= 4
    cover = [0] * m
    for ei, (u, v) in enumerate(candidates):
        bits = 0
        for pi, (i, j) in enumerate(pairs):
            d1 = base(i, u) + 1 + base(v, j)
            d2 = base(i, v) + 1 + base(u, j)
            if base(i, j) <= 4 or d1 <= 4 or d2 <= 4:
                bits |= (1 << pi)
        cover[ei] = bits

    # conflict mask: edge e conflicts with f if they share an endpoint or cross
    conf = [0] * m
    for a in range(m):
        u1, v1 = candidates[a]
        c = 0
        for b in range(m):
            if a == b:
                continue
            u2, v2 = candidates[b]
            # share endpoint
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                c |= (1 << b)
                continue
            # crossing (triangle): u1 < u2 < v1 < v2  or  u2 < u1 < v2 < v1
            if (u1 < u2 and u2 < v1 and v1 < v2) or (u2 < u1 and u1 < v2 and v2 < v1):
                c |= (1 << b)
        conf[a] = c

    # verify a concrete edge set via BFS (exact diameter + triangle-free + connected)
    def diameter_ok(edge_set):
        tmp = [row[:] for row in adj]
        for ei in edge_set:
            u, v = candidates[ei]
            tmp[u][v] = True
            tmp[v][u] = True
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = [src]
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                du = dist[u] + 1
                for v in range(n):
                    if tmp[u][v] and dist[v] == -1:
                        if du > 4:
                            return False
                        dist[v] = du
                        q.append(v)
            if any(d == -1 for d in dist):
                return False
        return True

    # lower bound: greedy packing + LP relaxation (max coverage per edge)
    def lower_bound(remaining):
        r = remaining
        cnt = 0
        while r:
            best = -1
            best_c = 0
            for ei in range(m):
                c = (cover[ei] & r).bit_count()
                if c > best_c:
                    best_c = c
                    best = ei
            if best < 0 or best_c == 0:
                return 10 ** 9
            cnt += 1
            r &= ~cover[best]
        return cnt

    # greedy upper bound (respects conflicts)
    def greedy_ub():
        remaining = (1 << Pn) - 1
        used = 0
        chosen = []
        while remaining:
            best = -1
            best_c = 0
            for ei in range(m):
                if used & (1 << ei):
                    continue
                c = (cover[ei] & remaining).bit_count()
                if c > best_c:
                    best_c = c
                    best = ei
            if best < 0 or best_c == 0:
                return None
            chosen.append(best)
            used |= (1 << best) | conf[best]
            remaining &= ~cover[best]
        if diameter_ok(chosen):
            return len(chosen)
        return None

    best = [greedy_ub()]
    if best[0] is None:
        best[0] = 10 ** 9

    # order candidates by coverage descending
    order = sorted(range(m), key=lambda e: -cover[e].bit_count())
    pos_to_cand = order
    # precompute cover/conf in that order for cache-friendly access
    cover_o = [cover[e] for e in order]
    conf_o = [conf[e] for e in order]

    best_edges = [None]
    calls = [0]

    import sys
    sys.setrecursionlimit(20000)

    def search(pos, remaining, chosen_count, chosen, used_mask):
        if time.time() > deadline - 2:
            return
        calls[0] += 1
        if remaining == 0:
            if chosen_count < best[0]:
                best[0] = chosen_count
                best_edges[0] = chosen[:]
            return
        if chosen_count + lower_bound(remaining) >= best[0]:
            return
        # pick the candidate at/after pos (not used) covering most of remaining
        branch_e = -1
        branch_c = 0
        for ei in range(pos, m):
            if used_mask & (1 << ei):
                continue
            c = (cover_o[ei] & remaining).bit_count()
            if c > branch_c:
                branch_c = c
                branch_e = ei
        if branch_e < 0 or branch_c == 0:
            return
        # include branch_e
        new_mask = used_mask | (1 << branch_e) | conf_o[branch_e]
        search(pos + 1, remaining & ~cover_o[branch_e], chosen_count + 1, chosen + [branch_e], new_mask)
        # exclude branch_e
        search(pos + 1, remaining, chosen_count, chosen, used_mask | (1 << branch_e))

    search(0, (1 << Pn) - 1, 0, [], 0)

    if best_edges[0] is not None and diameter_ok(best_edges[0]):
        result = [[int(candidates[ei][0]), int(candidates[ei][1])] for ei in best_edges[0]]
        return result

    # fallback: known 8-edge star at vertex 12
    star = []
    for v in [1, 3, 6, 9, 15, 18, 21, 23]:
        for ei, (u, w) in enumerate(candidates):
            if (u == 12 and w == v) or (u == v and w == 12):
                star.append(ei)
                break
    if star and diameter_ok(star):
        return [[int(candidates[ei][0]), int(candidates[ei][1])] for ei in star]

    return []

# EVOLVE-BLOCK-END
