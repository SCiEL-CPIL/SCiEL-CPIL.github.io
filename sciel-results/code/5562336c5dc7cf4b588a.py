"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    start = time.time()
    deadline = start + 80.0
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute a good variable ordering: highest degree first, then by index
    deg = [len(adj[i]) for i in range(n)]
    var_order = sorted(range(n), key=lambda x: (-deg[x], x))
    pos_in_order = [0] * n
    for i, v in enumerate(var_order):
        pos_in_order[v] = i

    # Precompute for each vertex, its neighbors sorted by position in var_order
    # (so we know which neighbors are "earlier" in the search)
    nbrs = [sorted(adj[i], key=lambda x: pos_in_order[x]) for i in range(n)]

    def verify(colors, k):
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # Check no bicolored cycles via union-find per color pair
        for i in range(k):
            for j in range(i + 1, k):
                verts = [x for x in range(n) if colors[x] in (i, j)]
                if len(verts) < 3:
                    continue
                parent = {v: v for v in verts}
                def find(x):
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
                vs = set(verts)
                for u in verts:
                    for v in adj[u]:
                        if v in vs and u < v:
                            if not union(u, v):
                                return False
        return True

    def local_search(k, max_iters=50000):
        """Local search: start with random valid proper coloring, then minimize bicolored cycles."""
        # First, get a proper k-coloring (greedy)
        colors = [-1] * n
        for v in var_order:
            used = set()
            for nb in adj[v]:
                if colors[nb] >= 0:
                    used.add(colors[nb])
            for c in range(k):
                if c not in used:
                    colors[v] = c
                    break
            if colors[v] < 0:
                return None  # Can't properly color with k colors greedily

        # Count bicolored cycles
        def count_cycles(col):
            count = 0
            for i in range(k):
                for j in range(i + 1, k):
                    verts = [x for x in range(n) if col[x] in (i, j)]
                    if len(verts) < 3:
                        continue
                    parent = {v: v for v in verts}
                    def find(x):
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
                    vs = set(verts)
                    for u in verts:
                        for v in adj[u]:
                            if v in vs and u < v:
                                if not union(u, v):
                                    count += 1
                                    break
                    if count > 0:
                        break
            return count

        best = colors[:]
        best_cost = count_cycles(best)
        if best_cost == 0:
            return best

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick random vertex and color
            v = random.randint(0, n - 1)
            old_c = colors[v]
            new_c = random.randint(0, k - 1)
            if new_c == old_c:
                continue
            # Check proper coloring
            valid = True
            for nb in adj[v]:
                if colors[nb] == new_c:
                    valid = False
                    break
            if not valid:
                continue
            colors[v] = new_c
            new_cost = count_cycles(colors)
            if new_cost <= best_cost:
                best_cost = new_cost
                best = colors[:]
                if best_cost == 0:
                    return best
            else:
                colors[v] = old_c

        return best if best_cost == 0 else None

    def solve_k(k):
        """Backtracking with union-find per color pair. Returns colors or None."""
        colors = [-1] * n
        # For each pair (c1,c2), maintain parent dict for DSU
        uf = {}
        for i in range(k):
            for j in range(i + 1, k):
                uf[(i, j)] = {}

        def dsu_find(pd, x):
            if x not in pd:
                pd[x] = x
                return x
            root = x
            while pd[root] != root:
                root = pd[root]
            return root

        def dsu_union(pd, x, y):
            rx = dsu_find(pd, x)
            ry = dsu_find(pd, y)
            if rx == ry:
                return False
            pd[rx] = ry
            return True

        nodes = [0]

        def bt(pos):
            if nodes[0] > 5000000:
                return False
            if time.time() > deadline - 3:
                return False
            if pos == n:
                return True

            v = var_order[pos]
            # Try colors, ordered by how many conflicts they cause (MRV-like)
            # Just try 0..k-1 but skip invalid ones
            for c in range(k):
                # Proper coloring check
                ok = True
                for nb in nbrs[v]:
                    if colors[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue

                # Bicolored cycle check: for each c2 != c, check neighbors with color c2
                conflict = False
                for c2 in range(k):
                    if c2 == c:
                        continue
                    nbs_c2 = [nb for nb in nbrs[v] if colors[nb] == c2]
                    if len(nbs_c2) < 2:
                        continue
                    pk = (min(c, c2), max(c, c2))
                    pd = uf[pk]
                    roots = set()
                    for nb in nbs_c2:
                        r = dsu_find(pd, nb)
                        if r in roots:
                            conflict = True
                            break
                        roots.add(r)
                    if conflict:
                        break
                if conflict:
                    continue

                # Assign
                colors[v] = c
                undo = []
                all_ok = True
                for c2 in range(k):
                    if c2 == c:
                        continue
                    nbs_c2 = [nb for nb in nbrs[v] if colors[nb] == c2]
                    if not nbs_c2:
                        continue
                    pk = (min(c, c2), max(c, c2))
                    pd = uf[pk]
                    for nb in nbs_c2:
                        rv = dsu_find(pd, v)
                        rn = dsu_find(pd, nb)
                        if rv == rn:
                            all_ok = False
                            break
                        undo.append((pk, rv, pd.get(rv)))
                        pd[rv] = rn
                    if not all_ok:
                        break

                if all_ok:
                    nodes[0] += 1
                    if bt(pos + 1):
                        return True

                # Undo
                for pk, rv, old_val in reversed(undo):
                    pd = uf[pk]
                    if old_val is None:
                        del pd[rv]
                    else:
                        pd[rv] = old_val
                colors[v] = -1

            return False

        if bt(0):
            return colors
        return None

    # Try increasing k with both backtracking and local search
    best_result = None
    best_k = n

    for k in range(3, 12):
        if time.time() > deadline - 10:
            break

        # Try backtracking with time limit
        if time.time() < deadline - 30:
            result = solve_k(k)
            if result is not None:
                if verify(result, k):
                    return [int(c) for c in result]

        # Try local search
        if time.time() < deadline - 15:
            ls_result = local_search(k, max_iters=30000)
            if ls_result is not None:
                if verify(ls_result, k):
                    if k < best_k:
                        best_k = k
                        best_result = ls_result[:]
                    return [int(c) for c in ls_result]

    # If we found something via local search, return it
    if best_result is not None:
        return [int(c) for c in best_result]

    # Fallback: greedy / simple patterns
    for k in range(3, 12):
        if time.time() > deadline - 5:
            break
        trial = [i % k for i in range(n)]
        if verify(trial, k):
            return [int(c) for c in trial]

    # Last resort: proper coloring with many colors (always valid)
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
