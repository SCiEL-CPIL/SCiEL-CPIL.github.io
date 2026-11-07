"""Combined: max-flow oracle + flow-guided greedy subdivision construction."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    degree = [len(adj[v]) for v in range(n)]

    # ---------- Bron-Kerbosch: true maximum clique ----------
    max_clique = []

    def bronk(R, P_set, X_set):
        nonlocal max_clique
        if time.time() > deadline:
            return
        if not P_set and not X_set:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        U = P_set | X_set
        if not U:
            return
        pivot = None
        max_inter = -1
        for u in U:
            inter = len(P_set & adj[u])
            if inter > max_inter:
                max_inter = inter
                pivot = u
        candidates = list(P_set - adj[pivot])
        for v in candidates:
            bronk(R | {v}, P_set & adj[v], X_set & adj[v])
            P_set.discard(v)
            X_set.add(v)
            if time.time() > deadline:
                return

    all_vertices = set(range(n))
    bronk(set(), all_vertices, set())
    if len(max_clique) < 2:
        max_clique = [edges[0][0], edges[0][1]]

    # ---------- Flow oracle: correctness check for a branch set ----------
    def flow_feasible(branches):
        k = len(branches)
        if k < 2:
            return False
        num_pairs = k * (k - 1) // 2
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2
        cap = [[0] * total_nodes for _ in range(total_nodes)]
        branch_set = set(branches)
        for v in range(n):
            cap[2 * v][2 * v + 1] = (k - 1) if v in branch_set else 1
        for u, v in edges:
            cap[2 * u + 1][2 * v] = 1
            cap[2 * v + 1][2 * u] = 1
        for b in branches:
            cap[S][2 * b] = k - 1
            cap[2 * b + 1][T] = k - 1

        def max_flow():
            flow = 0
            while True:
                parent = [-1] * total_nodes
                visited = [False] * total_nodes
                q = deque([S])
                visited[S] = True
                found = False
                while q:
                    u = q.popleft()
                    if u == T:
                        found = True
                        break
                    for v in range(total_nodes):
                        if not visited[v] and cap[u][v] > 0:
                            visited[v] = True
                            parent[v] = u
                            q.append(v)
                if not found:
                    break
                min_cap = float('inf')
                v = T
                while v != S:
                    u = parent[v]
                    if cap[u][v] < min_cap:
                        min_cap = cap[u][v]
                    v = u
                v = T
                while v != S:
                    u = parent[v]
                    cap[u][v] -= min_cap
                    cap[v][u] += min_cap
                    v = u
                flow += min_cap
            return flow

        return max_flow() >= num_pairs

    # ---------- Flow-guided greedy construction ----------
    def build_subdivision(branches, pool, max_depth=12, max_attempts=12):
        """Grow branch set by adding one vertex at a time, routing its new paths
        through the residual flow network (enforces vertex-disjoint interiors).
        Returns (branches, paths) or None."""
        branches = list(branches)
        k = len(branches)
        if k < 2:
            return None
        branch_set = set(branches)

        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2
        cap = [[0] * total_nodes for _ in range(total_nodes)]
        for v in range(n):
            cap[2 * v][2 * v + 1] = (k - 1) if v in branch_set else 1
        for u, v in edges:
            cap[2 * u + 1][2 * v] = 1
            cap[2 * v + 1][2 * u] = 1
        for b in branches:
            cap[S][2 * b] = k - 1
            cap[2 * b + 1][T] = k - 1
        # pre-route the initial complete set of paths
        total_pairs = k * (k - 1) // 2
        if max_flow_units(cap, S, T, total_nodes, total_pairs) < total_pairs:
            return None

        def add_branch(v):
            nonlocal k, branch_set
            k += 1
            branch_set.add(v)
            cap[2 * v][2 * v + 1] = k - 1
            cap[S][2 * v] = k - 1
            cap[2 * v + 1][T] = k - 1

        def route_new(v, cap):
            nonlocal k
            new_pairs = k - 1
            if max_flow_units(cap, S, T, total_nodes, new_pairs) < new_pairs:
                return False
            # extract the new-pair paths from residual flow
            paths = {}
            for b in branches:
                if b == v:
                    continue
                # follow flow out of b_out
                u = 2 * b + 1
                nxt = None
                for w in range(total_nodes):
                    if cap[u][w] < 1 and (cap[w][u] > 0):
                        nxt = w
                        break
                if nxt is None:
                    return False
                path = [b]
                cur = nxt
                while cur != 2 * v:
                    o = cur // 2
                    path.append(o)
                    cur = cur + 1 if cur % 2 == 0 else cur - 1
                    # advance to next in-node via a used edge
                    nxt2 = None
                    for w in range(total_nodes):
                        if cap[cur][w] < 1 and (cap[w][cur] > 0):
                            nxt2 = w
                            break
                    if nxt2 is None:
                        return False
                    cur = nxt2
                if cur != 2 * v:
                    return False
                paths[b] = path
            # verify each extracted path
            for b, p in paths.items():
                if p[0] != b or p[-1] != v:
                    return False
                for a, c in zip(p, p[1:]):
                    if (a, c) not in edge_set:
                        return False
                for x in p[1:-1]:
                    if x in branch_set:
                        return False
            for b, p in paths.items():
                all_paths[(b, v) if b < v else (v, b)] = list(p)
            return True

        all_paths = {}
        # record initial paths (we need them for the final answer)
        for a, b in combinations(branches, 2):
            all_paths[(a, b) if a < b else (b, a)] = [a, b]  # placeholder, will be replaced below

        # Actually extract initial paths properly by re-running flow from scratch is complex;
        # instead, if we grow to a final set, we re-extract everything at the end.
        # Keep a simple marker; final extraction below.

        candidates = [x for x in pool if x not in branch_set]
        current = list(branches)

        def extract_all_paths(bset):
            """Rebuild flow for bset and extract every pair path. Returns dict or None."""
            kk = len(bset)
            need = kk * (kk - 1) // 2
            S2 = 2 * n
            T2 = 2 * n + 1
            cap2 = [[0] * total_nodes for _ in range(total_nodes)]
            bset_s = set(bset)
            for v in range(n):
                cap2[2 * v][2 * v + 1] = (kk - 1) if v in bset_s else 1
            for u, v in edges:
                cap2[2 * u + 1][2 * v] = 1
                cap2[2 * v + 1][2 * u] = 1
            for b in bset:
                cap2[S2][2 * b] = kk - 1
                cap2[2 * b + 1][T2] = kk - 1
            if max_flow_units(cap2, S2, T2, total_nodes, need) < need:
                return None
            res = {}
            for b in bset:
                u = 2 * b + 1
                # collect all used out-edges from b_out
                out_edges = [w for w in range(total_nodes) if cap2[u][w] < 1 and (cap2[w][u] > 0)]
                for w in out_edges:
                    o = w // 2
                    if o == b:
                        continue
                    path = [b]
                    cur = w
                    while cur != 2 * o:
                        path.append(cur // 2)
                        cur = cur + 1 if cur % 2 == 0 else cur - 1
                        nxt2 = None
                        for x in range(total_nodes):
                            if cap2[cur][x] < 1 and (cap2[x][cur] > 0):
                                nxt2 = x
                                break
                        if nxt2 is None:
                            return None
                        cur = nxt2
                    if cur != 2 * o:
                        return None
                    res[(b, o) if b < o else (o, b)] = path
            if len(res) != need:
                return None
            return res

        def grow(bset, depth):
            if depth <= 0 or time.time() > deadline - 3:
                return list(bset)
            bset = list(bset)
            kk = len(bset)
            cands = [x for x in candidates if x not in set(bset)]
            # order by connectivity to current set
            cands.sort(key=lambda x: -sum(1 for y in bset if (x, y) in edge_set))
            for v in cands[:max_attempts]:
                nb = bset + [v]
                if not flow_feasible(nb):
                    continue
                deeper = grow(nb, depth - 1)
                if len(deeper) > len(bset):
                    return deeper
            return bset

        # iterative growth (avoid recursion depth issues)
        cur = list(branches)
        for _ in range(max_depth):
            if time.time() > deadline - 3:
                break
            kk = len(cur)
            cands = [x for x in candidates if x not in set(cur)]
            cands.sort(key=lambda x: -sum(1 for y in cur if (x, y) in edge_set))
            added = False
            for v in cands[:max_attempts]:
                nb = cur + [v]
                if flow_feasible(nb):
                    cur = nb
                    added = True
                    break
            if not added:
                break

        if len(cur) < 2:
            return None
        paths = extract_all_paths(cur)
        if paths is None:
            return None
        out_paths = []
        for a, b in combinations(cur, 2):
            out_paths.append(list(paths[(a, b) if a < b else (b, a)]))
        return cur, out_paths

    def max_flow_units(cap, S, T, total_nodes, need):
        flow = 0
        while flow < need:
            parent = [-1] * total_nodes
            visited = [False] * total_nodes
            q = deque([S])
            visited[S] = True
            found = False
            while q:
                u = q.popleft()
                if u == T:
                    found = True
                    break
                for v in range(total_nodes):
                    if not visited[v] and cap[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        q.append(v)
            if not found:
                break
            min_cap = float('inf')
            v = T
            while v != S:
                u = parent[v]
                if cap[u][v] < min_cap:
                    min_cap = cap[u][v]
                v = u
            v = T
            while v != S:
                u = parent[v]
                cap[u][v] -= min_cap
                cap[v][u] += min_cap
                v = u
            flow += min_cap
        return flow

    # ---------- Candidate pool ----------
    candidate_pool = sorted(range(n), key=lambda v: degree[v], reverse=True)

    best_branches = None
    best_paths = None
    best_size = 0

    def consider(bset):
        nonlocal best_branches, best_paths, best_size
        if len(bset) < 2:
            return
        res = build_subdivision(bset, candidate_pool,
                                max_depth=max(1, 14 - len(bset)),
                                max_attempts=10)
        if res is None:
            return
        b, p = res
        if len(b) > best_size:
            best_size = len(b)
            best_branches = list(b)
            best_paths = list(p)

    # Seed 1: max clique
    consider(max_clique)

    # Seed 2..: top-degree pairs
    top_deg = sorted(range(n), key=lambda v: degree[v], reverse=True)
    for si in range(min(6, len(top_deg))):
        if time.time() > deadline - 8:
            break
        sv = top_deg[si]
        seed = [sv]
        for w in top_deg:
            if w != sv and (sv, w) in edge_set:
                seed.append(w)
                break
        if len(seed) >= 2:
            consider(seed)

    # ---------- Fallback ----------
    if best_paths is None or len(best_paths) != len(list(combinations(best_branches, 2))):
        u, v = edges[0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]

    if len(branches_out) < 2:
        u, v = edges[0]
        branches_out = [int(u), int(v)]
        paths_out = [[int(u), int(v)]]

    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
