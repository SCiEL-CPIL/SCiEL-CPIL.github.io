<<<<<<< SEARCH
"""Rewritten program: fast bicolored-forest coloring via incremental union-find."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Degeneracy ordering (peeling) for fast backtracking
    order = _degeneracy_order(n, adj)

    deadline = time.time() + 80  # safety margin under 90s

    # Try increasing color counts; this graph should be solvable with 3 or 4.
    for k in range(2, 20):
        if time.time() > deadline - 3:
            break
        res = _solve(n, adj, order, k, deadline)
        if res is not None:
            return res

    # Fallback: greedy with a larger palette
    res = _greedy_fallback(n, adj, order)
    if res is not None:
        return res
    # Last resort (always valid: all distinct)
    return [int(i) for i in range(n)]


def _degeneracy_order(n, adj):
    deg = [len(adj[i]) for i in range(n)]
    remaining = set(range(n))
    order = []
    while remaining:
        v = min(remaining, key=lambda x: (deg[x], x))
        order.append(v)
        remaining.discard(v)
        for u in adj[v]:
            if u in remaining:
                deg[u] -= 1
    return order


def _solve(n, adj, order, k, deadline):
    color = [-1] * n
    color_count = [0] * k          # how many vertices have each color
    uf = {}                        # (c1,c2) -> parent list (size n)
    uf_edges = {}                  # (c1,c2) -> edge count in that subgraph

    def find(parents, x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(parents, x, y):
        rx, ry = find(parents, x), find(parents, y)
        if rx == ry:
            return False
        parents[rx] = ry
        return True

    def get_uf(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        key = (c1, c2)
        if key not in uf:
            uf[key] = list(range(n))
            uf_edges[key] = 0
        return uf[key], uf_edges[key]

    def try_assign(v, c):
        # Adjacency constraint: no neighbor may already have color c
        for u in adj[v]:
            if color[u] == c:
                return None
        # Incremental bicolored-forest check for each affected pair
        affected = set()
        for u in adj[v]:
            cu = color[u]
            if cu != -1:
                affected.add(cu)
        affected.add(c)
        affected.discard(c)

        # Tentatively assign
        color[v] = c
        color_count[c] += 1
        rollbacks = []
        ok = True
        for cu in affected:
            key1 = (c, cu) if c < cu else (cu, c)
            parents = uf.get(key1)
            if parents is None:
                parents = list(range(n))
                uf[key1] = parents
                uf_edges[key1] = 0
            # Add edges between v and its neighbors that have color cu or c
            # (v has color c; neighbors with color cu or c form the subgraph)
            for u in adj[v]:
                cuu = color[u]
                if cuu == c or cuu == cu:
                    # edge (v, u) in subgraph of pair {c, cu}
                    if not union(parents, v, u):
                        ok = False
                        break
                    uf_edges[key1] += 1
                    rollbacks.append((parents, v, u, key1))
            if not ok:
                break

        if not ok:
            # Undo
            for parents, a, b, key in rollbacks:
                # We can't easily undo union-find without storing parent changes.
                # Simpler: rebuild uf for affected pairs from scratch.
                pass
            # Rebuild affected pairs
            for cu in affected:
                key = (c, cu) if c < cu else (cu, c)
                _rebuild_uf(key, c, cu)
            color[v] = -1
            color_count[c] -= 1
            return None

        return rollbacks

    def _rebuild_uf(key, c1, c2):
        parents = list(range(n))
        uf[key] = parents
        uf_edges[key] = 0
        for v in range(n):
            cv = color[v]
            if cv == c1 or cv == c2:
                for u in adj[v]:
                    cu = color[u]
                    if (cu == c1 or cu == c2) and u > v:
                        if not union(parents, v, u):
                            uf_edges[key] = -1  # mark invalid
                            return
                        uf_edges[key] += 1

    def backtrack(idx):
        if time.time() > deadline - 2:
            return None
        if idx == n:
            return list(color)
        v = order[idx]
        # Try colors; order by count ascending (less crowded first) for diversity
        for c in range(k):
            # Quick check
            for u in adj[v]:
                if color[u] == c:
                    break
            else:
                # Assign and check
                color[v] = c
                color_count[c] += 1
                # Check all pairs involving c that have at least one neighbor
                affected = set()
                for u in adj[v]:
                    cu = color[u]
                    if cu != -1:
                        affected.add(cu)
                # Also need to check pairs {c, cu} where cu is any other used color
                # because edges within color c appear in every such pair.
                for cu in range(k):
                    if cu != c and color_count[cu] > 0:
                        affected.add(cu)

                ok = True
                for cu in affected:
                    key = (c, cu) if c < cu else (cu, c)
                    if key not in uf:
                        _rebuild_uf(key, c, cu)
                        if uf_edges[key] < 0:
                            ok = False
                            break
                    else:
                        # Rebuild to be safe (incremental union-find rollback is tricky)
                        _rebuild_uf(key, c, cu)
                        if uf_edges[key] < 0:
                            ok = False
                            break

                if ok:
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result

                color[v] = -1
                color_count[c] -= 1
        return None

    return backtrack(0)


def _greedy_fallback(n, adj, order):
    color = [-1] * n
    for v in order:
        used = set()
        for u in adj[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(50):
            if c not in used:
                color[v] = c
                break
    if -1 in color:
        return None
    return [int(c) for c in color]


# EVOLVE-BLOCK-END
=======
"""Rewritten program: fast bicolored-forest coloring via targeted cycle check."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    deadline = time.time() + 85

    for k in range(2, 15):
        if time.time() > deadline - 5:
            break
        res = _solve_fast(n, adj, k, deadline)
        if res is not None:
            return res

    res = _greedy_fallback(n, adj)
    if res is not None:
        return res
    return [int(i) for i in range(n)]


def _solve_fast(n, adj, k, deadline):
    color = [-1] * n
    order = _degeneracy_order(n, adj)

    def would_create_cycle(v, c, other):
        """Check if assigning color c to v creates a bicolored cycle with color 'other'.
        A cycle forms iff two neighbors of v with color 'other' are already connected
        in the subgraph induced by colors {c, other} (excluding v)."""
        other_neighbors = [u for u in adj[v] if color[u] == other]
        if len(other_neighbors) < 2:
            return False
        
        # BFS from first other-neighbor within {c, other} subgraph
        visited = set()
        queue = [other_neighbors[0]]
        visited.add(other_neighbors[0])
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                cnb = color[nb]
                if (cnb == c or cnb == other) and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        
        for nb in other_neighbors[1:]:
            if nb in visited:
                return True
        return False

    def backtrack(idx):
        if time.time() > deadline - 3:
            return None
        if idx == n:
            return list(color)

        v = order[idx]
        for cc in range(k):
            valid = True
            for u in adj[v]:
                if color[u] == cc:
                    valid = False
                    break
            if not valid:
                continue

            color[v] = cc
            ok = True
            for other in range(k):
                if other == cc:
                    continue
                # Only check if v has a neighbor with color 'other'
                has_other_neighbor = False
                for u in adj[v]:
                    if color[u] == other:
                        has_other_neighbor = True
                        break
                if not has_other_neighbor:
                    continue
                if would_create_cycle(v, cc, other):
                    ok = False
                    break

            if ok:
                result = backtrack(idx + 1)
                if result is not None:
                    return result

            color[v] = -1

        return None

    return backtrack(0)


def _degeneracy_order(n, adj):
    deg = [len(adj[i]) for i in range(n)]
    remaining = set(range(n))
    order = []
    while remaining:
        v = min(remaining, key=lambda x: (deg[x], x))
        order.append(v)
        remaining.discard(v)
        for u in adj[v]:
            if u in remaining:
                deg[u] -= 1
    return order


def _greedy_fallback(n, adj):
    color = [-1] * n
    for v in range(n):
        used = set()
        for u in adj[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(50):
            if c not in used:
                color[v] = c
                break
    if -1 in color:
        return None
    return [int(c) for c in color]


# EVOLVE-BLOCK-END
>>>>>>> REPLACE
