"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Precompute adjacency
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

def has_bicolored_cycle(coloring):
    """Check if any pair of colors induces a cycle using union-find."""
    n = N
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    for u, v in EDGES:
        cu, cv = coloring[u], coloring[v]
        if cu == cv:
            return True
        key = (min(cu, cv), max(cu, cv))
        # We need a DSU per color-pair. Use a dict.
        if key not in has_bicolored_cycle._dsu:
            has_bicolored_cycle._dsu[key] = (list(range(n)), [0]*n)
        par, rk = has_bicolored_cycle._dsu[key]
        # local find/union
        def f(x, par=par):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        ru, rv = f(u), f(v)
        if ru == rv:
            return True
        if rk[ru] < rk[rv]:
            ru, rv = rv, ru
        par[rv] = ru
        if rk[ru] == rk[rv]:
            rk[ru] += 1
    return False

has_bicolored_cycle._dsu = {}

def check_valid(coloring):
    """Full validation: adjacent differ and no bicolored cycle."""
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False
    # Check bicolored cycles
    dsu = {}
    for u, v in EDGES:
        cu, cv = coloring[u], coloring[v]
        if cu == cv:
            return False
        key = (min(cu, cv), max(cu, cv))
        if key not in dsu:
            dsu[key] = (list(range(N)), [0]*N)
        par, rk = dsu[key]
        def f(x, par=par):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        ru, rv = f(u), f(v)
        if ru == rv:
            return False
        if rk[ru] < rk[rv]:
            ru, rv = rv, ru
        par[rv] = ru
        if rk[ru] == rk[rv]:
            rk[ru] += 1
    return True

def backtracking_search(max_colors, deadline):
    """
    Backtracking search for a valid coloring using at most max_colors.
    Returns coloring list or None.
    """
    coloring = [-1] * N

    # Order vertices by degree descending for better pruning
    order = sorted(range(N), key=lambda v: -len(ADJ[v]))

    # For tracking bicolored cycles, we maintain a DSU per color-pair
    # dsu[(c1,c2)] = (parent, rank)
    dsu = {}

    def find(x, par):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def try_assign(v, c):
        """Try to assign color c to vertex v. Returns True if valid, False otherwise."""
        # Check adjacency: no neighbor can have color c
        for nb in ADJ[v]:
            if coloring[nb] == c:
                return False

        # Check bicolored cycles: for each neighbor with a different color,
        # the edge (v, nb) must not create a cycle in the subgraph of colors (c, coloring[nb])
        for nb in ADJ[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = (min(c, cn), max(c, cn))
            if key in dsu:
                par, rk = dsu[key]
                rv = find(v, par)
                rnb = find(nb, par)
                if rv == rnb:
                    return False  # Would create a bicolored cycle

        # All checks passed, commit
        coloring[v] = c
        for nb in ADJ[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = (min(c, cn), max(c, cn))
            if key not in dsu:
                dsu[key] = (list(range(N)), [0]*N)
            par, rk = dsu[key]
            rv = find(v, par)
            rnb = find(nb, par)
            if rk[rv] < rk[rnb]:
                rv, rnb = rnb, rv
            par[rnb] = rv
            if rk[rv] == rk[rnb]:
                rk[rv] += 1

        return True

    def undo_assign(v, c):
        """Undo the assignment of color c to vertex v."""
        # We need to undo the DSU unions. This is complex, so instead
        # we'll use a simpler approach: recompute DSU state from scratch
        # when backtracking. But that's expensive.
        # Alternative: use a persistent/undoable DSU.
        # For simplicity with 50 vertices, let's just mark v as uncolored
        # and handle DSU by rebuilding when needed.
        # Actually, let's use a different approach: store the DSU state
        # before each assignment for rollback.
        pass

    # Simpler approach: since N=50, we can afford to check validity
    # incrementally. Let's use a cleaner backtracking with full check.
    # But for efficiency, let's try a different strategy.

    # Actually, let's use a simpler but effective approach:
    # Assign vertices in order, and for each, try colors 0..max_colors-1.
    # Check constraints: (1) no same-color neighbor, (2) no bicolored cycle.
    # For (2), we maintain DSU per color-pair and support undo.

    # Let me rewrite with proper undo support using a stack.
    # For simplicity and given small N, let's just do a full check
    # at each step but only for the affected color-pairs.

    # Actually the cleanest approach for N=50:
    # Use recursive backtracking. At each vertex, try each color.
    # Check: for the new edge(s) created, verify no cycle in the
    # bicolored subgraph. We can do this by BFS/DFS in the subgraph
    # induced by the two colors, checking if adding this edge creates a cycle.

    # Let me use a different strategy: maintain the coloring, and when
    # assigning v->c, for each neighbor nb with color cn != c, check if
    # there's already a path from v to nb in the subgraph of colors {c, cn}
    # (excluding the direct edge). If yes, it would form a cycle.

    # For N=50, a BFS per edge check is fine.

    def would_create_cycle(v, c, cn):
        """Check if adding edge (v, nb) with colors (c, cn) creates a cycle.
        i.e., is there a path from v to nb in the subgraph induced by colors {c,cn}
        using only already-assigned vertices, not using the direct edge?"""
        # BFS from v in subgraph of colors {c, cn}, avoiding nb as endpoint
        # Actually: check if nb is reachable from v in the {c,cn}-subgraph
        # without using the direct edge (v,nb). Since we're adding (v,nb),
        # a cycle exists iff there's another path from v to nb.
        # BFS from v, but don't traverse the edge (v,nb) directly.
        # Actually simpler: BFS from v in the {c,cn} subgraph, see if we reach nb
        # via any path of length >= 2.
        # Or: check if v and nb are already connected in the {c,cn} subgraph
        # (before adding this edge). If yes, adding the edge creates a cycle.

        # Build adjacency for colors {c, cn} among assigned vertices
        target_colors = {c, cn}
        visited = set()
        visited.add(v)
        queue = [v]
        while queue:
            cur = queue.pop(0)
            for nb2 in ADJ[cur]:
                if nb2 == v and cur == nb:  # skip the direct edge back
                    continue
                if coloring[nb2] != -1 and coloring[nb2] in target_colors and nb2 not in visited:
                    if nb2 == nb:
                        # Found nb via a path of length >= 2
                        return True
                    visited.add(nb2)
                    queue.append(nb2)
        return False

    # Hmm, this BFS approach might be too slow if called many times.
    # Let me think of a better way.

    # Alternative: Use the DSU approach with proper undo.
    # Since we're doing DFS backtracking, we can save the DSU state
    # before each assignment and restore it on backtrack.

    # For N=50 and max_colors up to ~8, the search tree is manageable.
    # Let me use a simpler approach: just try all assignments and
    # do a full check at the end, but with good pruning.

    # Actually, the most practical approach for N=50:
    # 1. Try small k values (3, 4, 5, ...)
    # 2. For each k, do backtracking with pruning:
    #    - No two adjacent same color
    #    - No bicolored cycle (checked incrementally with DSU + undo)

    # Let me implement with DSU and undo stack.

    # Reset
    coloring = [-1] * N
    dsu = {}  # key -> (parent, rank)
    undo_stack = []  # list of (key, v, nb, par_before, rk_before) for rollback

    def find(x, par):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def dfs(idx):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        v = order[idx]
        for c in range(max_colors):
            # Check: no neighbor has color c
            valid = True
            for nb in ADJ[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue

            # Check: for each neighbor with a different assigned color,
            # adding edge (v,nb) must not create a bicolored cycle
            cycle_created = False
            for nb in ADJ[v]:
                cn = coloring[nb]
                if cn == -1 or cn == c:
                    continue
                key = (min(c, cn), max(c, cn))
                if key in dsu:
                    par, rk = dsu[key]
                    rv = find(v, par)
                    rnb = find(nb, par)
                    if rv == rnb:
                        cycle_created = True
                        break
            if cycle_created:
                continue

            # All good, assign and recurse
            coloring[v] = c
            # Record DSU changes for undo
            saved_changes = []
            for nb in ADJ[v]:
                cn = coloring[nb]
                if cn == -1 or cn == c:
                    continue
                key = (min(c, cn), max(c, cn))
                if key not in dsu:
                    dsu[key] = (list(range(N)), [0]*N)
                    saved_changes.append((key, 'new', None, None))
                par, rk = dsu[key]
                rv = find(v, par)
                rnb = find(nb, par)
                if rk[rv] < rk[rnb]:
                    rv, rnb = rnb, rv
                saved_changes.append((key, 'union', rnb, rk[rv]))
                par[rnb] = rv
                if rk[rv] == rk[rnb]:
                    rk[rv] += 1

            if dfs(idx + 1):
                return True

            # Undo
            coloring[v] = -1
            for change in reversed(saved_changes):
                key, typ, a, b = change
                if typ == 'new':
                    del dsu[key]
                elif typ == 'union':
                    par, rk = dsu[key]
                    # a is the child root, b is the rank before increment
                    # We need to undo: par[a] was set to rv, and rk[rv] was incremented
                    # This is tricky. Let me save more info.
                    # Actually, let me just save the full DSU state for simplicity.
                    pass

            # The undo is complex. Let me use a simpler approach:
            # Save the entire dsu dict before assignment, restore after.
            # For N=50, this is acceptable.

        return False

    # The undo approach above is messy. Let me use a cleaner method:
    # Save and restore the entire dsu state at each level.

    dsu = {}

    def dfs2(idx):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        v = order[idx]
        for c in range(max_colors):
            # Check adjacency
            valid = True
            for nb in ADJ[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue

            # Check bicolored cycles using current dsu
            cycle_created = False
            for nb in ADJ[v]:
                cn = coloring[nb]
                if cn == -1 or cn == c:
                    continue
                key = (min(c, cn), max(c, cn))
                if key in dsu:
                    par, rk = dsu[key]
                    rv = find(v, par)
                    rnb = find(nb, par)
                    if rv == rnb:
                        cycle_created = True
                        break
            if cycle_created:
                continue

            # Save state
            old_dsu = {k: (list(par), list(rk)) for k, (par, rk) in dsu.items()}
            old_color = coloring[v]

            # Assign
            coloring[v] = c
            for nb in ADJ[v]:
                cn = coloring[nb]
                if cn == -1 or cn == c:
                    continue
                key = (min(c, cn), max(c, cn))
                if key not in dsu:
                    dsu[key] = (list(range(N)), [0]*N)
                par, rk = dsu[key]
                rv = find(v, par)
                rnb = find(nb, par)
                if rk[rv] < rk[rnb]:
                    rv, rnb = rnb, rv
                par[rnb] = rv
                if rk[rv] == rk[rnb]:
                    rk[rv] += 1

            if dfs2(idx + 1):
                return True

            # Restore
            dsu.clear()
            dsu.update(old_dsu)
            coloring[v] = old_color

        return False

    if dfs2(0):
        return coloring
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s budget

    # Try increasing number of colors
    for k in range(3, 15):
        if time.time() > deadline:
            break
        result = backtracking_search(k, deadline)
        if result is not None:
            # Verify
            if check_valid(result):
                return [int(x) for x in result]

    # Fallback: return seed coloring (50 colors, always valid)
    return list(range(N))

# EVOLVE-BLOCK-END
