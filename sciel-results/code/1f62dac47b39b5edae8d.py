"""Rewritten program: exact backtracking + forward checking + incremental DSU.
Searches for the minimum number of colors k (3,4,5,6) for the pinned degree-4 graph.
Constraint: adjacent vertices differ AND every 2-color induced subgraph is a forest."""

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    import time
    deadline = time.time() + 80.0   # safety margin below the 90s kill

    n = P['n']
    adj = [[] for _ in range(n)]
    for a, b in P['edges']:
        adj[a].append(b)
        adj[b].append(a)
    deg = [len(adj[i]) for i in range(n)]

    # Static vertex ordering: high degree first, then (static) largest neighbor color-set.
    static_order = list(range(n))
    static_order.sort(key=lambda i: (-deg[i], -max((len(adj[j]) for j in adj[i]), default=0)))
    pos = {v: i for i, v in enumerate(static_order)}

    # --- Incremental DSU with undo, one instance per color-pair (compact arrays) ---
    def make_pair_dsu():
        parent = list(range(n))
        size = [1] * n
        rank = [0] * n

        def find(x):
            r = x
            while parent[r] != r:
                r = parent[r]
            # path compression
            while parent[x] != r:
                nx = parent[x]
                parent[x] = r
                x = nx
            return r

        def unite(a, b):
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return False   # would form a cycle
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        return parent, size, rank, find, unite

    def _solve_with_k(k):
        # Per-pair DSU: index p = c1*k + c2 (c1 < c2)
        dsu = [make_pair_dsu() for _ in range(k * (k - 1) // 2)]

        def pidx(c1, c2):
            if c1 > c2:
                c1, c2 = c2, c1
            return c1 * k + c2 - c2 * (c2 + 1) // 2

        def pair_cycle(c1, c2, u, v):
            if c1 == c2:
                return False
            parent, size, rank, find, unite = dsu[pidx(c1, c2)]
            if find(u) == find(v):
                return True
            return not unite(u, v)

        # Undo stack: (pair_index, a, b, merged, rank_before)
        undo = []

        def undo_all():
            while undo:
                pi, a, b, merged, rank_before = undo.pop()
                parent, size, rank, find, unite = dsu[pi]
                if merged:
                    # restore: recompute parents by re-uniting original components is hard;
                    # instead we stored nothing per-edge, so use a global rebuild on backtrack.
                    pass

        # Because per-edge undo of union-find is fiddly, we use a simpler, robust approach:
        # recompute the DSU state from scratch for the current partial assignment is too slow.
        # Instead, we store full snapshots of parent/rank for each pair only at the top of recursion
        # via a copy-on-write stack. To keep it fast, we snapshot the *changed* roots.
        # Simpler and still fast: keep a list of (pair, rootA, rootB, rankA_before) and undo by
        # reversing the parent pointer we set. We do NOT path-compress in find during the search
        # to make undo trivial (find without compression).
        raise RuntimeError("replaced below")

    # The above sketch is replaced by a clean implementation:
    def _solve_with_k(k):
        # Pair DSU with NO path compression so undo is a single pointer revert.
        parent = [[i for i in range(n)] for _ in range(k * (k - 1) // 2)]
        rank = [[0] * n for _ in range(k * (k - 1) // 2)]

        def pidx(c1, c2):
            if c1 > c2:
                c1, c2 = c2, c1
            return c1 * k + c2 - c2 * (c2 + 1) // 2

        def find(pi, x):
            while parent[pi][x] != x:
                x = parent[pi][x]
            return x

        def unite(pi, a, b):
            ra = find(pi, a)
            rb = find(pi, b)
            if ra == rb:
                return None
            if rank[pi][ra] < rank[pi][rb]:
                ra, rb = rb, ra
            parent[pi][rb] = ra
            if rank[pi][ra] == rank[pi][rb]:
                rank[pi][ra] += 1
            return (pi, rb, ra, rank[pi][ra])

        color = [-1] * n
        # Dynamic neighbor color-set size for ordering (updated as colors assigned).
        nset = [0] * n
        # count how many neighbors already colored per vertex
        colored_nbr = [0] * n

        # order: recompute a heuristic key on the fly; use static order as base,
        # but at each step pick the unassigned vertex with max (colored_nbr, deg).
        # We precompute static_order and iterate, but choose the "most constrained"
        # unassigned vertex by (colored_nbr, deg) for stronger pruning.
        unassigned = set(range(n))

        def pick_vertex():
            best = -1
            best_key = (-1, -1)
            for v in unassigned:
                key = (colored_nbr[v], deg[v])
                if key > best_key:
                    best_key = key
                    best = v
            return best

        def backtrack(depth):
            if depth == n:
                return True
            if time.time() > deadline:
                return False
            v = pick_vertex()
            unassigned.discard(v)
            nbrs = adj[v]
            # colors forbidden by colored neighbors
            forbidden = set()
            for u in nbrs:
                cu = color[u]
                if cu != -1:
                    forbidden.add(cu)
            candidates = [c for c in range(k) if c not in forbidden]
            # forward check: for each candidate, ensure no cycle with colored same-color-pair neighbors
            ok_candidates = []
            for c in candidates:
                good = True
                for u in nbrs:
                    cu = color[u]
                    if cu != -1 and cu != c:
                        pi = pidx(c, cu)
                        if find(pi, v) == find(pi, u):
                            good = False
                            break
                if good:
                    ok_candidates.append(c)
            if not ok_candidates:
                unassigned.add(v)
                return False
            # order candidates: prefer colors already used by many neighbors (less new pairs)
            ok_candidates.sort(key=lambda c: -sum(1 for u in nbrs if color[u] == c))
            for c in ok_candidates:
                color[v] = c
                # record union changes
                changes = []
                for u in nbrs:
                    cu = color[u]
                    if cu != -1 and cu != c:
                        pi = pidx(c, cu)
                        ch = unite(pi, v, u)
                        if ch is None:
                            # cycle formed -> should not happen (pre-checked) but be safe
                            break
                        changes.append(ch)
                # update colored_nbr for neighbors
                for u in nbrs:
                    colored_nbr[u] += 1
                if backtrack(depth + 1):
                    return True
                # undo
                for u in nbrs:
                    colored_nbr[u] -= 1
                for (pi, rb, ra, rank_before) in reversed(changes):
                    parent[pi][rb] = rb
                    rank[pi][ra] = rank_before
                color[v] = -1
                unassigned.add(v)
            return False

        return backtrack(0), color

    best_color = None
    best_k = None
    for k in (3, 4, 5, 6):
        if time.time() > deadline - 5.0:
            break
        ok, color = _solve_with_k(k)
        if ok:
            # verify
            if _verify(color):
                best_color = color
                best_k = k
                break
        # even if not ok, keep color as a fallback if it happens to be valid
        if best_color is None and _verify(color):
            best_color = color
            best_k = k

    if best_color is None:
        # last resort: always-valid 50-color identity (valid, score 50)
        best_color = list(range(n))
        best_k = n

    return [int(c) for c in best_color]


def _verify(color):
    n = P['n']
    # adjacency
    for a, b in P['edges']:
        if color[a] == color[b]:
            return False
    # bicolored cycle check via DSU per pair
    from collections import defaultdict
    # build per-pair edge list
    pairs = defaultdict(list)
    for a, b in P['edges']:
        c1, c2 = color[a], color[b]
        if c1 == c2:
            return False
        if c1 > c2:
            c1, c2 = c2, c1
        pairs[(c1, c2)].append((a, b))
    for (c1, c2), edges in pairs.items():
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            parent[ru] = rv
    return True
