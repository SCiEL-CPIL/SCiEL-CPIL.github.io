"""Biclique edge-partition: greedy set-cover + local-search polish."""
import time

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_EDGES = P['edges']
_N = P['n']


def _valid(part):
    seen = set()
    for left, right in part:
        if not left or not right:
            return False
        for a in left:
            for b in right:
                e = (a, b) if a < b else (b, a)
                if e in seen:
                    return False
                seen.add(e)
    return len(seen) == len(_EDGES)


def _make_solution(part):
    out = []
    for left, right in part:
        L = [int(x) for x in left]
        R = [int(x) for x in right]
        out.append([L, R])
    return out


def _greedy_partition(remaining):
    """Greedy: repeatedly pick the biclique covering the most remaining edges."""
    # adjacency restricted to remaining edges
    adj = {v: set() for v in range(_N)}
    for a, b in remaining:
        adj[a].add(b)
        adj[b].add(a)

    part = []
    rem = set(remaining)
    while rem:
        best = None
        best_cov = -1
        # candidate left = a vertex; right = a subset of its remaining neighbors
        # Try each vertex as a hub, right = all remaining neighbors, then try to
        # grow left by adding vertices adjacent to every vertex in right.
        for hub in list(adj.keys()):
            if not adj[hub]:
                continue
            right = set(adj[hub])
            # left: vertices (other than hub) adjacent to all in right
            left = {hub}
            for v in adj[hub]:
                if v != hub:
                    # v can be in left if it is adjacent to all of right
                    ok = True
                    for w in right:
                        if w != v and w not in adj[v]:
                            ok = False
                            break
                    if ok:
                        left.add(v)
            # count covered edges
            cov = 0
            for a in left:
                for b in right:
                    if a == b:
                        continue
                    e = (a, b) if a < b else (b, a)
                    if e in rem:
                        cov += 1
            if cov > best_cov:
                best_cov = cov
                best = (left, right)
        # If nothing found (shouldn't happen), fall back to a star
        if best is None or best_cov == 0:
            # pick an uncovered edge
            a, b = next(iter(rem))
            best = ({a}, {b})
            best_cov = 1
        left, right = best
        part.append((left, right))
        # remove covered edges
        to_remove = set()
        for a in left:
            for b in right:
                if a == b:
                    continue
                e = (a, b) if a < b else (b, a)
                if e in rem:
                    to_remove.add(e)
        rem -= to_remove
        # update adj
        for a, b in to_remove:
            adj[a].discard(b)
            adj[b].discard(a)
    return part


def _try_reduce(part, remaining, deadline):
    """Local search: try to merge two bicliques into one, or replace a biclique
    with a larger one that absorbs edges from others."""
    adj = {v: set() for v in range(_N)}
    for a, b in remaining:
        adj[a].add(b)
        adj[b].add(a)
    part = [ (list(l), list(r)) for l, r in part ]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(part)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(n):
                if i == j:
                    continue
                # Try to merge biclique j into biclique i
                Li, Ri = part[i]
                Lj, Rj = part[j]
                # new biclique must be a valid biclique (all cross edges in graph)
                # and must cover all edges of i and j
                candL = set(Li) | set(Lj)
                candR = set(Ri) | set(Rj)
                # check all cross pairs are edges
                ok = True
                for a in candL:
                    for b in candR:
                        if a == b:
                            continue
                        if b not in adj[a]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # check it covers all edges of i and j
                covered = set()
                for a in candL:
                    for b in candR:
                        if a == b:
                            continue
                        e = (a, b) if a < b else (b, a)
                        covered.add(e)
                # edges of i
                for a in Li:
                    for b in Ri:
                        e = (a, b) if a < b else (b, a)
                        if e not in covered:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                for a in Lj:
                    for b in Rj:
                        e = (a, b) if a < b else (b, a)
                        if e not in covered:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # merge: remove j, update i
                part[i] = (list(candL), list(candR))
                part.pop(j)
                improved = True
                break
            if improved:
                break
    return part


def construct():
    deadline = time.time() + 85.0
    remaining = _EDGES[:]
    part = _greedy_partition(remaining)
    # polish
    if time.time() < deadline:
        part = _try_reduce(part, remaining, deadline)
    # ensure validity
    if not _valid(part):
        # fallback: star partition
        part = []
        for u in range(_N):
            nbrs = [v for a, v in _EDGES if a == u] + [a for a, b in _EDGES if b == u]
            if nbrs:
                part.append(([u], list(set(nbrs))))
    return _make_solution(part)
# EVOLVE-BLOCK-END
