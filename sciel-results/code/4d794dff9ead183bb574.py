"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t0 = time.time()
    DEADLINE = 80.0

    n = int(P['n'])
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    adj_list = [list(s) for s in adj]

    # ---------- Max clique (branch-and-bound, Tomita order) ----------
    def max_clique():
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        pos = {v: i for i, v in enumerate(order)}
        R = [0] * n
        for i, v in enumerate(order):
            R[i] = v
        best = []
        best_size = 0
        stack = []
        Pset = set(range(n))
        stack.append((list(Pset), 0))
        while stack:
            Pcur, size = stack.pop()
            if not Pcur:
                continue
            if size > best_size:
                best_size = size
                best = [R[i] for i in range(size)]
            for i in range(len(Pcur) - 1, -1, -1):
                v = Pcur[i]
                if size + i + 1 <= best_size:
                    break
                R[size] = v
                newP = [u for u in Pcur[:i] if u in adj[v]]
                if newP:
                    stack.append((newP, size + 1))
        return best

    # ---------- Max flow (Dinic) ----------
    class Dinic:
        __slots__ = ('n', 'g', 'lvl', 'it')

        def __init__(self, n):
            self.n = n
            self.g = [[] for _ in range(n)]
            self.lvl = [0] * n
            self.it = [0] * n

        def add_edge(self, u, v, c):
            self.g[u].append([v, c, len(self.g[v])])
            self.g[v].append([u, 0, len(self.g[u]) - 1])

        def bfs(self, s, t):
            from collections import deque
            lvl = [-1] * self.n
            lvl[s] = 0
            q = deque([s])
            g = self.g
            while q:
                u = q.popleft()
                for e in g[u]:
                    if e[1] > 0 and lvl[e[0]] < 0:
                        lvl[e[0]] = lvl[u] + 1
                        q.append(e[0])
            self.lvl = lvl
            return lvl[t] >= 0

        def dfs(self, u, t, f):
            if u == t:
                return f
            g = self.g
            lvl = self.lvl
            it = self.it
            i = it[u]
            while i < len(g[u]):
                e = g[u][i]
                if e[1] > 0 and lvl[u] + 1 == lvl[e[0]]:
                    d = self.dfs(e[0], t, f if f < e[1] else e[1])
                    if d > 0:
                        e[1] -= d
                        g[e[0]][e[2]][1] += d
                        return d
                i += 1
                it[u] = i
            return 0

        def max_flow(self, s, t):
            flow = 0
            INF = float('inf')
            while self.bfs(s, t):
                self.it = [0] * self.n
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    def try_branches(B):
        k = len(B)
        need = k * (k - 1) // 2
        if need == 0:
            return None
        Bset = set(B)
        INF = 10 ** 9
        N = 2 * k + 2
        S = 2 * k
        T = 2 * k + 1
        idx = {v: i for i, v in enumerate(B)}
        din = Dinic(N)
        for i, v in enumerate(B):
            din.add_edge(2 * i, 2 * i + 1, 1)          # in -> out (cap 1)
            din.add_edge(S, 2 * i + 1, 1)              # source -> out
            din.add_edge(2 * i, T, 1)                  # in -> sink
        for v in B:
            i = idx[v]
            for w in adj[v]:
                if w in Bset:
                    din.add_edge(2 * i + 1, 2 * idx[w], INF)   # out -> in (internal)
        flow = din.max_flow(S, T)
        if flow < need:
            return None
        # decompose flow into paths
        paths = []
        for u in range(N):
            for e in din.g[u]:
                if e[1] < 0:                          # carries flow
                    v = e[0]
                    amt = -e[1]
                    if u == S:
                        start = B[idx_inv[2 * idx.get(v, -1)]] if False else None
                        # v is an out node (2*i+1)
                        start = B[(v - 1) // 2]
                    elif v == T:
                        # u is an in node (2*i)
                        end = B[u // 2]
                        start = None
                    else:
                        start = None
                    if start is None:
                        continue
                    # walk from start to end following flow edges
                    path = [start]
                    cur = start
                    cur_out = 2 * idx[cur] + 1
                    while cur != end:
                        nxt = None
                        for e2 in din.g[cur_out]:
                            if e2[1] < 0:
                                nxt = e2[0]
                                break
                        if nxt is None:
                            break
                        nxt_v = B[nxt // 2]
                        path.append(nxt_v)
                        cur = nxt_v
                        cur_out = 2 * idx[cur] + 1
                    if path and path[0] == start and path[-1] == end and len(path) >= 2:
                        paths.append(path)
        return paths

    # build idx_inv helper not needed; inline above uses idx
    # (redefine try_branches cleanly to avoid the messy walk)
    def try_branches(B):
        k = len(B)
        need = k * (k - 1) // 2
        if need == 0:
            return None
        Bset = set(B)
        INF = 10 ** 9
        N = 2 * k + 2
        S = 2 * k
        T = 2 * k + 1
        idx = {v: i for i, v in enumerate(B)}
        din = Dinic(N)
        for i, v in enumerate(B):
            din.add_edge(2 * i, 2 * i + 1, 1)
            din.add_edge(S, 2 * i + 1, 1)
            din.add_edge(2 * i, T, 1)
        for v in B:
            i = idx[v]
            for w in adj[v]:
                if w in Bset:
                    din.add_edge(2 * i + 1, 2 * idx[w], INF)
        flow = din.max_flow(S, T)
        if flow < need:
            return None
        # map node id -> branch vertex
        node2v = {}
        for i, v in enumerate(B):
            node2v[2 * i] = v
            node2v[2 * i + 1] = v
        paths = []
        # For each branch, its out-node has flow leaving toward sink-side.
        # Walk: start at branch b -> out(b); follow flow edges until in(c) -> c.
        for b in B:
            i = idx[b]
            cur_out = 2 * i + 1
            # find all flow edges leaving cur_out
            for e in din.g[cur_out]:
                if e[1] < 0:
                    nxt = e[0]
                    path = [b]
                    cur = nxt
                    while cur != T:
                        vv = node2v[cur]
                        path.append(vv)
                        # move to out of vv
                        cur = 2 * idx[vv] + 1
                        # follow the single flow edge out of cur
                        adv = None
                        for e2 in din.g[cur]:
                            if e2[1] < 0:
                                adv = e2[0]
                                break
                        if adv is None:
                            break
                        cur = adv
                    if len(path) >= 2 and path[-1] != b:
                        paths.append(path)
        if len(paths) != need:
            return None
        return paths

    # ---------- Search ----------
    best_paths = None
    best_B = None
    best_size = 0

    # fallback: a single edge
    if P['edges']:
        u, v = P['edges'][0]
        best_paths = [[int(u), int(v)]]
        best_B = [int(u), int(v)]
        best_size = 2

    cliques = max_clique()
    # collect all maximal cliques via same routine is heavy; instead use greedy + subsets
    # Build candidate branch sets: start from max clique, then try removing vertices.
    candidates = []
    if len(cliques) >= 2:
        candidates.append(cliques)
    # Also try greedy independent-ish large sets? Keep to cliques/subsets.
    # Enumerate subsets of the max clique by size descending.
    mc = list(cliques)
    m = len(mc)
    for size in range(m, 1, -1):
        if time.time() > DEADLINE:
            break
        for sub in combinations(mc, size):
            sub = list(sub)
            if best_size and size <= best_size:
                break
            if time.time() > DEADLINE:
                break
            paths = try_branches(sub)
            if paths is not None:
                best_paths = paths
                best_B = sub
                best_size = size
                break
        if time.time() > DEADLINE:
            break

    # Also try a couple of other maximal cliques found by greedy restarts
    def greedy_clique():
        cand = list(range(n))
        clique = []
        while cand and time.time() < DEADLINE:
            # pick vertex with max degree within cand
            best_v = max(cand, key=lambda v: sum(1 for u in cand if u in adj[v]))
            clique.append(best_v)
            cand = [u for u in cand if u != best_v and u in adj[best_v]]
        return clique

    for _ in range(40):
        if time.time() > DEADLINE:
            break
        gc = greedy_clique()
        if len(gc) <= best_size:
            continue
        gmc = list(gc)
        m = len(gmc)
        for size in range(m, 1, -1):
            if time.time() > DEADLINE:
                break
            if size <= best_size:
                break
            for sub in combinations(gmc, size):
                sub = list(sub)
                if time.time() > DEADLINE:
                    break
                paths = try_branches(sub)
                if paths is not None:
                    best_paths = paths
                    best_B = sub
                    best_size = size
                    break
            if time.time() > DEADLINE:
                break

    # sanitize output
    B = [int(x) for x in best_B]
    paths_out = []
    seen = set()
    for p in best_paths:
        p = [int(x) for x in p]
        key = frozenset((p[0], p[-1]))
        if key in seen:
            continue
        seen.add(key)
        paths_out.append(p)

    if len(B) < 2 or len(paths_out) < 1:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    return {'branches': B, 'paths': paths_out}
# EVOLVE-BLOCK-END
