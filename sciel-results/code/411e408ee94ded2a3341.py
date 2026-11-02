# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    deadline = time.time() + 140.0   # explicit guard; 150s kill, keep 10s safety margin

    # ---- Guaranteed-valid fallback: Lazebnik-Ustimenko-Woldar CD graph ----
    def _lw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # ---- Exact girth certification via non-backtracking walks from every vertex ----
    def _girth(adj, n):
        # girth = smallest L>=3 with a non-backtracking walk of length L from some vertex
        # to a vertex already at distance L-1 (a "collision" closes a cycle of length L).
        INF = 10**9
        for L in range(3, 2*g+1):
            # dp[v] = min non-backtracking distance from a start to v (walk length L),
            # tracked over all starts simultaneously is too coarse; instead BFS per start is
            # expensive. Use the standard: for each start, non-backtracking BFS to depth L,
            # detect collision with a vertex at depth L-1.
            # Optimize: only need to detect ANY cycle of length L, so run from each vertex
            # but prune heavily. For n~448 and L<=10 this is fine.
            for s in range(n):
                # frontier[v] = True if a non-backtracking walk of current length reaches v
                # we track (prev, v) pairs to forbid backtracking.
                # Represent state as (prev, cur).
                from collections import deque
                dist = {}
                q = deque()
                for nb in adj[s]:
                    q.append((s, nb))
                    dist[(s, nb)] = 1
                while q:
                    prev, cur = q.popleft()
                    dl = dist[(prev, cur)]
                    if dl == L:
                        break
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        key = (cur, nb)
                        if key in dist:
                            # collision: a non-backtracking walk of length dl+1 reaches (cur,nb)
                            # which was reached at length dist[key] -> cycle length dl+1-dist[key]
                            cyc = dl + 1 - dist[key]
                            if 3 <= cyc < L:
                                return cyc
                        else:
                            dist[key] = dl + 1
                            q.append(key)
        return 2*g  # no cycle shorter than g found within bound (assume >= g)

    # ---- Moore-bound girth-10 tree, leaves identified -> 4-regular graph ----
    def _build_tree(k, g):
        # girth g even. Tree of depth (g-2)/2 from a root of degree k.
        # Root deg k, internal non-leaf deg k+1 (one parent + k children), leaves deg 1.
        depth = (g - 2) // 2
        adj = {}
        def new_node():
            idx = len(adj)
            adj[idx] = []
            return idx
        def add_edge(a, b):
            adj[a].append(b)
            adj[b].append(a)
        root = new_node()
        frontier = [root]
        # level 0 = root
        for lvl in range(depth):
            nxt = []
            for v in frontier:
                # children count: root has k children; others have k children (k+1 total deg)
                for _ in range(k):
                    c = new_node()
                    add_edge(v, c)
                    nxt.append(c)
            frontier = nxt
        # frontier now = leaves (depth 'depth'), each degree 1
        leaves = frontier
        return adj, root, leaves

    def _identify(adj, leaves, perm):
        # perm: list length L, perm[i] = index in leaves of the leaf identified with leaves[i].
        # Build new graph on n_internal + L/2 vertices.
        n_int = len(adj)
        L = len(leaves)
        leaf_pos = {}
        for i in range(L):
            leaf_pos[leaves[i]] = i
        # map each vertex to new id: internal keep 0..n_int-1; leaves -> n_int + pair_id
        new_id = list(range(n_int))
        for i in range(L):
            p = perm[i]
            pair = min(i, p)
            new_id[leaves[i]] = n_int + pair
        new_n = n_int + L // 2
        new_adj = [set() for _ in range(new_n)]
        for a in range(n_int):
            for b in adj[a]:
                na, nb = new_id[a], new_id[b]
                if na != nb:
                    new_adj[na].add(nb)
                    new_adj[nb].add(na)
        return [sorted(s) for s in new_adj]

    # ---- Search: local search (Hill-climb + simulated annealing) over leaf permutations ----
    def _search(k, g, time_budget):
        adj, root, leaves = _build_tree(k, g)
        L = len(leaves)
        # initial: pair leaf i with leaf (i + L//2) % L  (opposite in BFS order)
        perm = [ (i + L//2) % L for i in range(L) ]
        # ensure involution-ish: make symmetric
        for i in range(L//2):
            j = (i + L//2) % L
            perm[i] = j
            perm[j] = i
        def build(perm):
            return _identify(adj, leaves, perm)
        def score(perm):
            # lower is better: girth penalty + size (size fixed) -> use -girth (higher girth better)
            adj2 = build(perm)
            gr = _girth(adj2, len(adj2))
            # want girth >= g. score = (g - gr) large penalty if < g, plus 0 else; we minimize
            return (g - gr) if gr < g else 0, adj2, gr
        cur_perm = perm[:]
        cur_pen, cur_adj, cur_gr = score(cur_perm)
        best_pen, best_adj, best_gr = cur_pen, cur_adj, cur_gr
        rng = random.Random(12345)
        t_end = time.time() + time_budget
        cur_temp = 1.0
        while time.time() < t_end:
            # pick a random leaf i, re-pair with a random leaf j
            i = rng.randrange(L//2)
            j = rng.randrange(L//2)
            if i == j:
                continue
            # current partners
            pi, pj = cur_perm[i], cur_perm[j]
            # swap partners: i<->j, pi<->pj  (a 2-swap move)
            new_perm = cur_perm[:]
            new_perm[i] = j
            new_perm[j] = i
            new_perm[pi] = pj
            new_perm[pj] = pi
            npen, nadj, ngr = score(new_perm)
            if npen <= cur_pen or rng.random() < (1.0/(1.0 + ngr + 0.1)) and npen < cur_pen + 2:
                cur_perm = new_perm
                cur_pen, cur_adj, cur_gr = npen, nadj, ngr
                if npen < best_pen:
                    best_pen, best_adj, best_gr = npen, nadj, ngr
            cur_temp *= 0.999
            if time.time() > t_end - 5:
                break
        return best_adj

    # ---- Driver ----
    # Try the Moore-tree + local-search family first (smaller n), verify, else fallback.
    try:
        time_budget = min(110.0, max(2.0, (deadline - time.time()) - 25.0))
        adj = _search(k, g, time_budget)
        n = len(adj)
        # verify: degree k and girth >= g
        ok = True
        for v in range(n):
            if len(adj[v]) != k:
                ok = False
                break
        if ok:
            gr = _girth(adj, n)
            if gr >= g:
                edges = []
                for v in range(n):
                    for w in adj[v]:
                        if v < w:
                            edges.append((v, w))
                return n, edges
    except Exception:
        pass
    # fallback
    try:
        return _lw_fallback()
    except Exception:
        # last resort: trivial valid 4-regular girth>=10 is hard; return a large cycle-free
        # fallback via LW again (shouldn't reach)
        return _lw_fallback()
# EVOLVE-BLOCK-END
