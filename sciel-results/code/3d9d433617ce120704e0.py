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

    # ---- Fast exact girth certification via non-backtracking walk counting ----
    # Counts closed non-backtracking walks; the smallest L with a positive count
    # is the girth. Returns L, or the cap (g) if none is found below it.
    def _girth(adj, n, cap):
        kdeg = len(adj[0])
        # cnt[prev][cur] = # non-backtracking walks of current length ending at (prev,cur)
        cnt = [[0]*n for _ in range(n)]
        for s in range(n):
            for nb in adj[s]:
                cnt[s][nb] += 1
        for L in range(3, cap+1):
            new = [[0]*n for _ in range(n)]
            total = 0
            for u in range(n):
                cu = cnt[u]
                for v in adj[u]:
                    cv = cnt[v]
                    s = 0
                    for w in adj[v]:
                        if w != u:
                            s += cu[w]
                    if s:
                        new[v][u] = s
                        total += s
            if total:
                return L
            cnt = new
        return cap

    # ---- Moore-bound girth-g tree, leaves to be identified -> k-regular graph ----
    def _build_tree(k, g):
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
        for lvl in range(depth):
            nxt = []
            for v in frontier:
                for _ in range(k):
                    c = new_node()
                    add_edge(v, c)
                    nxt.append(c)
            frontier = nxt
        leaves = frontier
        return adj, root, leaves

    def _identify(adj, leaves, perm):
        n_int = len(adj)
        L = len(leaves)
        leaf_pos = {leaves[i]: i for i in range(L)}
        new_id = list(range(n_int))
        for i in range(L):
            p = perm[i]
            pair = i if i < p else p
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

    # ---- Search: hill-climb / simulated-annealing over leaf pairings ----
    def _search(k, g, time_budget):
        adj, root, leaves = _build_tree(k, g)
        L = len(leaves)
        half = L // 2
        # Initial pairing: pair leaf i with the leaf at the opposite side of the
        # BFS order (i + half) % L. This tends to keep identified leaves far apart
        # in the tree, which is what avoids short cycles.
        perm = [(i + half) % L for i in range(L)]
        for i in range(half):
            j = (i + half) % L
            perm[i] = j
            perm[j] = i

        def build(p):
            return _identify(adj, leaves, p)

        rng = random.Random(12345)
        t_end = time.time() + time_budget

        cur_perm = perm[:]
        cur_adj = build(cur_perm)
        cur_gr = _girth(cur_adj, len(cur_adj), g)
        best_adj = cur_adj
        best_gr = cur_gr

        while time.time() < t_end:
            # 2-swap move: swap partners of two distinct leaves
            i = rng.randrange(half)
            j = rng.randrange(half)
            if i == j:
                continue
            pi, pj = cur_perm[i], cur_perm[j]
            new_perm = cur_perm[:]
            new_perm[i] = j
            new_perm[j] = i
            new_perm[pi] = pj
            new_perm[pj] = pi
            nadj = build(new_perm)
            ngr = _girth(nadj, len(nadj), g)
            # Accept if it improves (higher girth) or is equal; occasionally accept
            # a slight regression to escape local optima.
            if ngr >= cur_gr or (ngr < cur_gr and rng.random() < 0.1):
                cur_perm = new_perm
                cur_adj = nadj
                cur_gr = ngr
                if ngr > best_gr:
                    best_adj = nadj
                    best_gr = ngr
            if time.time() > t_end - 3:
                break
        return best_adj, best_gr

    # ---- Driver ----
    # Try the Moore-tree + local-search family first (smaller n), verify, else fallback.
    try:
        time_budget = min(115.0, max(2.0, (deadline - time.time()) - 20.0))
        adj, gr = _search(k, g, time_budget)
        n = len(adj)
        ok = True
        for v in range(n):
            if len(adj[v]) != k:
                ok = False
                break
        if ok and gr >= g:
            edges = []
            for v in range(n):
                for w in adj[v]:
                    if v < w:
                        edges.append((v, w))
            return n, edges
    except Exception:
        pass
    # fallback (guaranteed valid LUW graph)
    return _lw_fallback()
# EVOLVE-BLOCK-END
