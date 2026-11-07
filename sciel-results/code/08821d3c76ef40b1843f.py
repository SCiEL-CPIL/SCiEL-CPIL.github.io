import time
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

# Precompute: for each vertex, its neighbors sorted (for ordering)
deg = [len(a) for a in adj]

def _build_pair_edges(k):
    """For a given k, return for each color pair (c1<c2) the list of edges
    (u,v) where the two endpoints are constrained to be colored with c1 and c2
    in some order. Actually we build dynamically; this helper not used."""
    return None

def _is_valid(coloring, k):
    """Check adjacent differ and each 2-color induced subgraph is a forest."""
    # adjacent differ
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False
    # bicolored forest per pair
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            # collect edges whose both endpoints use c1 or c2 and are assigned
            # Actually induced subgraph on vertices of colors {c1,c2}: include
            # all edges between those vertices.
            # Use union-find to detect cycle.
            parent = list(range(N))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in EDGES:
                cu = coloring[u]
                cv = coloring[v]
                if cu in (c1, c2) and cv in (c1, c2):
                    ru = find(u)
                    rv = find(v)
                    if ru == rv:
                        return False
                    parent[ru] = rv
    return True

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Precompute a good vertex ordering: highest degree first for search
    order = list(range(N))
    order.sort(key=lambda v: (-deg[v], v))

    # We'll search over k from 3 upward. For each k, do backtracking with
    # incremental bicolored-cycle detection.
    best = None

    # A strong baseline: greedy + local improvement to get a valid coloring.
    # First, get a trivially valid coloring (50 colors) then improve.
    # But we want few colors. Let's just run the search.

    # Helper: for a partial assignment, maintain per-pair union-find state.
    # We'll do DFS assigning vertices in `order`.

    # For incremental cycle detection, we maintain for each pair (a,b) a
    # union-find over vertices. Adding an edge (u,v) with colors (cu,cv) where
    # {cu,cv} == {a,b} must not create a cycle in that pair's graph.

    # Represent pair index
    def pair_idx(a, b):
        if a > b:
            a, b = b, a
        return a * N + b  # unique since a,b in 0..k-1 <= 49, N=50 -> a*50+b unique for a<=b? a*50+b with a<b could collide? a1*50+b1=a2*50+b2 => a1=a2,b1=b2 since b<50. Yes unique.

    for k in range(3, 15):
        if time.time() - t0 > DEADLINE - 5:
            break
        # Precompute pair structures
        # For each pair (a,b), we need union-find. We'll create lazily.
        # To make it fast, we precompute for each pair the list of edges that
        # could belong to it given the current coloring of endpoints.
        # But coloring changes. Instead, during DFS, when we assign a vertex,
        # we check all its already-assigned neighbors.

        # We maintain:
        #   color[v] = assigned color or -1
        #   For each pair, a DSU. We'll store DSU as dict pair->parent list.
        #   To rollback, we need to undo union operations. We'll keep a stack
        #   of (pair, ru, rv) for each union we perform, and also record the
        #   edges we added per pair for rollback of "edge count" if needed.
        #   Actually DSU union is only needed for cycle detection; we can
        #   rollback by storing the parent change.

        # Simpler: since N=50 small, we can, at each assignment, for each
        # affected pair, recompute whether adding the new edges creates a
        # cycle by running a quick cycle check on that pair's edge set.
        # But that's O(E) per pair per step. With ~75 pairs and 50 steps,
        # 75*50*E ~ 75*50*90 = 337500 per leaf, too slow for backtracking.

        # Better: maintain DSU per pair with rollback.
        # We'll store for each pair a parent array and a rank, and a stack
        # of changes.

        # Let's implement DSU with rollback per pair, created on demand.

        # We'll store:
        #   dsu = {} mapping pair_key -> [parent_list, rank_list, stack]
        #   stack entries: (child_root, parent_root) meaning we set parent[child]=parent_root
        #   To rollback one step, pop and restore parent[child]=child.

        # When we assign vertex v with color cv, for each neighbor u already
        # assigned color cu:
        #   if cu == cv: conflict (adjacent same) -> invalid
        #   else: pair (min(cu,cv), max(cu,cv)); add edge (u,v) to that pair's
        #         DSU. If union finds same root -> cycle -> invalid.

        # We need to remember which unions we did for this assignment to
        # rollback when backtracking. We'll push a marker.

        color = [-1] * N
        dsu = {}
        # Stack of operations: each entry either ('mark', depth) or
        # ('union', pair_key, child, parent)
        op_stack = []

        def dsu_get(pkey):
            if pkey not in dsu:
                dsu[pkey] = [list(range(N)), [0] * N, []]
            return dsu[pkey]

        def dsu_find(state, x):
            parent = state[0]
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def dsu_union(state, a, b):
            parent = state[0]
            rank = state[1]
            ra = dsu_find(state, a)
            rb = dsu_find(state, b)
            if ra == rb:
                return False  # cycle
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            state[2].append((rb, ra, rank[ra] - (1 if rank[ra] > rank[rb] else 0)))
            # Simpler: store (rb, ra) and we'll also need to decrement rank if
            # it was increased. Let's store (rb, ra, rank_ra_before).
            return True

        # Let me redo union to store enough for rollback.
        def dsu_union2(state, a, b):
            parent = state[0]
            rank = state[1]
            ra = dsu_find(state, a)
            rb = dsu_find(state, b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            increased = (rank[ra] == rank[rb])
            if increased:
                rank[ra] += 1
            state[2].append((rb, ra, increased))
            return True

        def dsu_rollback(state, count):
            parent = state[0]
            rank = state[1]
            for _ in range(count):
                rb, ra, increased = state[2].pop()
                parent[rb] = rb
                if increased:
                    rank[ra] -= 1

        def backtrack(idx):
            nonlocal best
            if time.time() - t0 > DEADLINE:
                return False
            if idx == N:
                # all assigned, valid by construction
                best = list(color)
                return True
            v = order[idx]
            # Try colors 0..k-1
            for cv in range(k):
                # Check neighbors already assigned
                ok = True
                unions_done = []  # list of (pkey, state, count)
                for u in adj[v]:
                    if color[u] != -1:
                        cu = color[u]
                        if cu == cv:
                            ok = False
                            break
                        pkey = pair_idx(cu, cv)
                        state = dsu_get(pkey)
                        # union u and v
                        if not dsu_union2(state, u, v):
                            # cycle
                            ok = False
                            # rollback this union? dsu_union2 returns False
                            # without pushing, so nothing to rollback.
                            break
                        unions_done.append((pkey, state, 1))
                if not ok:
                    # rollback any unions done before the failing one
                    for pkey, state, cnt in unions_done:
                        dsu_rollback(state, cnt)
                    continue
                color[v] = cv
                # recurse
                if backtrack(idx + 1):
                    return True
                color[v] = -1
                # rollback unions
                for pkey, state, cnt in unions_done:
                    dsu_rollback(state, cnt)
            return False

        # Reset structures for this k
        color = [-1] * N
        dsu = {}
        if backtrack(0):
            if best is not None:
                return [int(x) for x in best]

    # Fallback: return a valid coloring with more colors if search didn't
    # find one (should not happen, but safe).
    # Greedy: assign each vertex the smallest color not used by neighbors,
    # then verify and fix.
    coloring = [0] * N
    # Simple greedy in order
    for v in range(N):
        used = set(coloring[u] for u in adj[v] if coloring[u] != -1)
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    # This greedy may create bicolored cycles; verify and if invalid, fall back
    # to 50 distinct colors.
    if _is_valid(coloring, max(coloring) + 1):
        return [int(x) for x in coloring]
    return [int(i) for i in range(N)]

# Quick self-test at import (cheap)
if __name__ == "__main__":
    r = construct()
    assert isinstance(r, list)
    assert len(r) == 50
    assert all(isinstance(x, int) for x in r)
    print("colors used:", len(set(r)))
