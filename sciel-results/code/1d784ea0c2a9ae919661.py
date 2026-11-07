"""Clique-subdivision search: max-flow feasibility + incremental greedy growth."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _max_clique(n, adj):
    """Bron-Kerbosch with pivot, returns a maximum clique as a list."""
    best = [0]
    best_set = set()
    for v in range(n):
        best_set.add(v)
        best_set.discard(v)
    for v in range(n):
        best_set.add(v)
        best_set.discard(v)
    best_set = set()

    def bk(R, P, X):
        if len(R) + len(P) <= len(best_set):
            return
        if not P and not X:
            if len(R) > len(best_set):
                best_set.clear()
                best_set.update(R)
            return
        PX = P | X
        if not PX:
            return
        pivot = max(PX, key=lambda x: len(adj[x] & P))
        for v in list(P):
            if v in adj[pivot]:
                continue
            R2 = R | {v}
            bk(R2, P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)

    bk(set(), set(range(n)), set())
    return list(best_set)


def _dinic_solve(n, branches, adj):
    """Return (feasible, paths) for the branch set.
    paths: list of (u, v, [vertices]) for each unordered pair, or None."""
    k = len(branches)
    need = k * (k - 1) // 2
    if need == 0:
        return True, []

    idx = {b: i for i, b in enumerate(branches)}
    # node ids
    S = 0
    T = 1
    def inid(v): return 2 + v
    def outid(v): return 2 + n + v
    NN = 2 + 2 * n

    g = [[] for _ in range(NN)]
    def add_edge(u, v, c):
        g[u].append([v, c, len(g[v])])
        g[v].append([u, 0, len(g[u]) - 1])

    # vertex capacities
    for v in range(n):
        if v in idx:
            add_edge(inid(v), outid(v), need)   # branch: high capacity
        else:
            add_edge(inid(v), outid(v), 1)       # interior: capacity 1

    # source / sink
    for b in branches:
        add_edge(S, inid(b), k - 1)
        add_edge(outid(b), T, k - 1)

    # graph edges
    for u in range(n):
        for v in adj[u]:
            if u < v:
                add_edge(outid(u), inid(v), need)
                add_edge(outid(v), inid(u), need)

    # Dinic
    flow = 0
    level = [0] * NN
    it = [0] * NN
    def bfs():
        for i in range(NN):
            level[i] = -1
        level[S] = 0
        q = deque([S])
        while q:
            u = q.popleft()
            for e in g[u]:
                v, c, _ = e
                if c > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    if v == T:
                        return True
                    q.append(v)
        return level[T] >= 0

    def dfs(u, f):
        if u == T:
            return f
        while it[u] < len(g[u]):
            e = g[u][it[u]]
            v, c, rev = e
            if c > 0 and level[v] == level[u] + 1:
                ret = dfs(v, f if f < c else c)
                if ret:
                    e[1] -= ret
                    g[v][rev][1] += ret
                    return ret
            it[u] += 1
        return 0

    while bfs():
        for i in range(NN):
            it[i] = 0
        while True:
            pushed = dfs(S, 10 ** 9)
            if not pushed:
                break
            flow += pushed
            if flow >= need:
                break
    if flow < need:
        return False, None

    # extract paths
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append((branches[i], branches[j]))
    # map each source edge S->in(b) to the pair it serves
    # We track which (a,b) uses the unit on S->in(a).
    # Build list of (a,b) and assign to source edges of a.
    from collections import defaultdict
    pairs_by_a = defaultdict(list)
    for a, b in pairs:
        pairs_by_a[a].append(b)

    path_for_edge = {}   # (S, inid(a)) -> (a,b)
    for a in branches:
        for b in pairs_by_a[a]:
            # find the forward edge S->inid(a)
            for e in g[S]:
                if e[0] == inid(a) and e[1] > 0:
                    path_for_edge[(S, inid(a))] = (a, b)
                    break

    # For each source edge with residual capacity, there is a unit of flow.
    # We need the original capacity to know how many units. Reconstruct by
    # scanning residual: flow on S->in(a) = orig_cap - residual. orig_cap = k-1.
    # Walk from each such unit.
    paths = {}
    # For each a, we have (k-1) units. Reconstruct by repeated path extraction
    # using a separate residual snapshot is complex; instead we extract by
    # tracing from S through edges that currently carry flow.
    # Simpler: for each (a,b) pair, find a path in the residual-flow graph.
    # We do a BFS in the "flow graph" (edges with positive flow) from a to b.

    # Build flow adjacency: for each original directed edge, flow = reverse residual.
    flow_adj = [[] for _ in range(NN)]
    # iterate all forward edges (those added as real edges have reverse with cap 0 initially)
    # We stored edges as [v, c, rev]. Flow on forward = g[v][rev][1].
    for u in range(NN):
        for e in g[u]:
            v, c, rev = e
            rev_cap = g[v][rev][1]
            if rev_cap > 0:  # there is flow on u->v
                flow_adj[u].append(v)

    def trace(a, b):
        # BFS in flow_adj from inid(a) to outid(b)
        parent = {inid(a): None}
        q = deque([inid(a)])
        while q:
            u = q.popleft()
            if u == outid(b):
                # reconstruct
                verts = []
                cur = outid(b)
                while cur is not None:
                    if cur >= 2:
                        node = (cur - 2) % n
                        if not verts or verts[-1] != node:
                            verts.append(node)
                    p = parent[cur]
                    cur = p
                verts.reverse()
                return verts
            for v in flow_adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        return None

    result = []
    for a, b in pairs:
        verts = trace(a, b)
        if verts is None:
            return False, None
        result.append((a, b, verts))
    return True, result


def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)

    def time_left():
        return DEADLINE - (time.time() - t_start)

    best_branches = None
    best_paths = None

    def record(branches, paths):
        nonlocal best_branches, best_paths
        if best_paths is None or len(branches) > len(best_branches):
            best_branches = list(branches)
            best_paths = list(paths)

    def greedy_grow(seed):
        """Incrementally grow a branch set from seed, re-solving flow each step."""
        branches = list(seed)
        # verify seed feasible
        ok, paths = _dinic_solve(n, branches, adj)
        if not ok:
            return
        record(branches, paths)
        while time_left() > 2.0:
            k = len(branches)
            candidates = [v for v in range(n) if v not in set(branches)]
            # order candidates by degree descending
            candidates.sort(key=lambda v: len(adj[v]), reverse=True)
            added = False
            for v in candidates:
                if time_left() < 2.0:
                    break
                trial = branches + [v]
                ok, paths = _dinic_solve(n, trial, adj)
                if ok:
                    branches = trial
                    record(branches, paths)
                    added = True
                    break
            if not added:
                break

    # Start from maximum clique
    clique = _max_clique(n, adj)
    if len(clique) >= 2:
        greedy_grow(clique)

    # Try multiple random seeds to escape local optima
    random.seed(12345)
    all_v = list(range(n))
    attempts = 0
    while time_left() > 6.0 and attempts < 40:
        attempts += 1
        size = random.randint(2, min(6, n))
        seed = random.sample(all_v, size)
        # ensure seed itself is feasible before growing (flow test)
        ok, paths = _dinic_solve(n, seed, adj)
        if ok:
            greedy_grow(seed)

    # Fallback: guarantee a valid solution (at least one edge)
    if best_paths is None:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [(u, v, [u, v])]

    # Build final dict
    branches = [int(b) for b in best_branches]
    paths_list = []
    for a, b, verts in best_paths:
        path = [int(x) for x in verts]
        if path[0] != a:
            path = [a] + path
        if path[-1] != b:
            path = path + [b]
        paths_list.append(path)

    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
