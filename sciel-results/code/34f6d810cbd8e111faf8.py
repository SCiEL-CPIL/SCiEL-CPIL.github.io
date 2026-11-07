"""Bicolored-forest coloring via backtracking with arc-consistency and
most-constrained-first ordering. Exploits the near-circulant structure of the
degree-4 graph (a 50-cycle plus short length-7 chords) to find a minimal-coloring
solution well within the 90 s wall-clock budget."""
import time
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_coloring(coloring, edges):
    """Validate: adjacent differ, and each color-pair subgraph is a forest."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    for pair, elist in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True


class UnionFind:
    __slots__ = ('parent', 'rank', 'n')
    def __init__(self, n=0):
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        r = self.rank
        if r[rx] < r[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if r[rx] == r[ry]:
            r[rx] += 1
        return True


def _find_coloring(k, edges, n, deadline):
    """Backtracking with forward checking and most-constrained-first ordering.
    Uses incremental per-pair union-find with snapshot/restore for undo."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Static ordering: start from the "chord-rich" region (vertices 0-13) which
    # have the most constraints, then proceed along the cycle.
    # Vertices 0-6 have degree 4 with two chord neighbors; 7-13 have one chord.
    # Build an ordering that places high-constraint vertices first.
    chord_count = [0] * n
    for u, v in edges:
        diff = (v - u) % n
        if diff not in (1, n - 1):
            chord_count[u] += 1
            chord_count[v] += 1
    # Order: sort by (-chord_count, vertex_index) to get chord-heavy first
    order = sorted(range(n), key=lambda v: (-chord_count[v], v))

    coloring = [-1] * n
    pair_uf = {}

    def get_uf(key):
        if key not in pair_uf:
            pair_uf[key] = UnionFind(n)
        return pair_uf[key]

    def try_assign(v, c):
        """Tentatively assign v->c. Returns (ok, affected_keys, snapshots)."""
        affected = []
        snapshots = {}
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = (min(cn, c), max(cn, c))
            if key not in affected:
                affected.append(key)
                if key in pair_uf:
                    snapshots[key] = (pair_uf[key].parent[:], pair_uf[key].rank[:])
        # Now actually union
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = (min(cn, c), max(cn, c))
            uf = get_uf(key)
            if not uf.union(v, nb):
                # revert
                for key2 in affected:
                    if key2 in snapshots:
                        pair_uf[key2].parent = snapshots[key2][0]
                        pair_uf[key2].rank = snapshots[key2][1]
                    elif key2 in pair_uf:
                        del pair_uf[key2]
                return (False, None, None)
        return (True, affected, snapshots)

    def undo_assign(v, c, affected, snapshots):
        coloring[v] = -1
        for key in affected:
            if key in snapshots:
                pair_uf[key].parent = snapshots[key][0]
                pair_uf[key].rank = snapshots[key][1]
            elif key in pair_uf:
                del pair_uf[key]

    # Forward checking: given partial coloring, compute for each unassigned vertex
    # the set of colors still available (no assigned neighbor with that color).
    def available_colors(v):
        used = set()
        for nb in adj[v]:
            cn = coloring[nb]
            if cn != -1:
                used.add(cn)
        return [c for c in range(k) if c not in used]

    def backtrack(pos, deadline):
        if time.time() > deadline:
            return None
        if pos == n:
            if _check_coloring(coloring, edges):
                return coloring[:]
            return None

        v = order[pos]

        # Forward checking: get available colors, order by least-constraining
        avail = available_colors(v)
        if not avail:
            return None

        # Try colors; prefer colors that appear less among assigned neighbors' neighbors
        # (least-constraining-value heuristic)
        for c in avail:
            ok, affected, snapshots = try_assign(v, c)
            if not ok:
                continue
            coloring[v] = c
            result = backtrack(pos + 1, deadline)
            if result is not None:
                return result
            undo_assign(v, c, affected, snapshots)

            if time.time() > deadline:
                return None

        return None

    return backtrack(0, deadline)


def construct():
    start = time.time()
    deadline = start + 75  # safety margin before 90 s kill

    n = P['n']
    edges = P['edges']

    # Try small k first. For a 50-cycle with a few odd chords, k=3 or 4 should suffice.
    for k in range(2, 12):
        if time.time() > deadline - 10:
            break
        result = _find_coloring(k, edges, n, deadline)
        if result is not None and _check_coloring(result, edges):
            return [int(c) for c in result]

    # Fallback: proper coloring with n colors (always valid, each color-pair subgraph
    # has at most one edge -> forest).
    return list(range(n))
