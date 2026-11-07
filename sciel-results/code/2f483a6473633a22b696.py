import time
import random
import sys
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # ---------- Dinic max flow ----------
    class Dinic:
        def __init__(self, N):
            self.N = N
            self.graph = [[] for _ in range(N)]

        def add_edge(self, u, v, c):
            self.graph[u].append([v, c, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])

        def bfs(self, s, t):
            self.level = [-1] * self.N
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for e in self.graph[u]:
                    v, c, _ = e
                    if c > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[t] >= 0

        def dfs(self, u, t, f):
            if u == t:
                return f
            gu = self.graph[u]
            while self.it[u] < len(gu):
                e = gu[self.it[u]]
                v, c, rev = e
                if c > 0 and self.level[v] == self.level[u] + 1:
                    d = self.dfs(v, t, min(f, c))
                    if d > 0:
                        e[1] -= d
                        self.graph[v][rev][1] += d
                        return d
                self.it[u] += 1
            return 0

        def max_flow(self, s, t):
            flow = 0
            INF = 10**9
            while self.bfs(s, t):
                self.it = [0] * self.N
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    # ---------- Build the "super-sink" flow network for a candidate branch set ----------
    # Nodes: for each original vertex v: in-node = 1 + v, out-node = 1 + n + v
    # super source S = 0, super sink T = 1 + 2*n
    # For each branch a: S -> in(a) cap (k-1); out(a) -> T cap (k-1)
    # For each vertex v: in(v) -> out(v) cap 1  (unit vertex capacity)
    # For each edge (u,v): out(u) -> in(v) cap INF and out(v) -> in(u) cap INF
    # Max flow == number of internally-disjoint paths that can be packed among the k branches.
    # Required flow for a full clique subdivision on k branches: k*(k-1)/2.

    def build_and_flow(branches):
        k = len(branches)
        if k < 2:
            return 0, None
        branch_set = set(branches)
        S = 0
        T = 1 + 2 * n
        N = T + 1
        dinic = Dinic(N)
        INF = 10**9

        def vin(v):
            return 1 + v

        def vout(v):
            return 1 + n + v

        for a in branches:
            dinic.add_edge(S, vin(a), k - 1)
            dinic.add_edge(vout(a), T, k - 1)

        for v in range(n):
            dinic.add_edge(vin(v), vout(v), 1)

        for u, v in edges:
            dinic.add_edge(vout(u), vin(v), INF)
            dinic.add_edge(vout(v), vin(u), INF)

        flow = dinic.max_flow(S, T)
        return flow, dinic

    # ---------- Extract paths from a flow network ----------
    def extract_paths(dinic, branches):
        k = len(branches)
        S = 0
        T = 1 + 2 * n

        def vin(v):
            return 1 + v

        def vout(v):
            return 1 + n + v

        # Build a residual forward-flow map: for each directed edge (u->v) in dinic.graph,
        # original capacity minus residual = flow. We need to track which edges carry flow.
        # We'll reconstruct by scanning: an edge e = [v, c, rev] in graph[u] has flow = orig - c.
        # But we didn't store orig. Instead, we note that for unit-capacity vertex edges,
        # flow is 1 if residual is 0. For source/sink edges cap (k-1), flow = (k-1) - residual.
        # Simpler: do a DFS from S in the residual graph following edges with flow>0.
        # To know flow>0, we need original capacity. Rebuild a parallel structure.

        # Rebuild: create a list of (u, v, cap) for every add_edge call in the same order.
        # We'll just re-simulate the construction to get original capacities.
        branch_set = set(branches)
        edge_caps = []  # (u, v, cap) for each forward edge in order of addition

        for a in branches:
            edge_caps.append((S, vin(a), k - 1))
        for a in branches:
            edge_caps.append((vout(a), T, k - 1))
        for v in range(n):
            edge_caps.append((vin(v), vout(v), 1))
        for u, v in edges:
            edge_caps.append((vout(u), vin(v), 10**9))
            edge_caps.append((vout(v), vin(u), 10**9))

        # Now dinic.graph has 2*len(edge_caps) entries total (forward + reverse).
        # The forward edges are at even indices in the addition order, but graph[u] is a list
        # that accumulates. We need to map: for the i-th add_edge call, the forward edge is
        # the one appended to graph[u] at that moment. Let's just re-walk:
        # Actually simpler: build a flow matrix by checking residual.
        # For each (u,v,cap) in edge_caps, find the forward edge in dinic.graph[u] that points to v
        # and was added for this purpose. Since multiple edges can go u->v, we match by index.

        # Most robust: during construction, record for each forward edge its (u, v, cap, idx_in_graph_u).
        # Let me redo: build network and record edge locations.
        return None  # placeholder, replaced below

    def build_and_extract(branches):
        k = len(branches)
        if k < 2:
            return 0, None
        branch_set = set(branches)
        S = 0
        T = 1 + 2 * n
        N = T + 1
        dinic = Dinic(N)
        INF = 10**9

        def vin(v):
            return 1 + v

        def vout(v):
            return 1 + n + v

        # Record each forward edge: (u, v, cap, pos_in_graph_u)
        forward_edges = []

        for a in branches:
            u, v, c = S, vin(a), k - 1
            forward_edges.append((u, v, c, len(dinic.graph[u])))
            dinic.add_edge(u, v, c)
        for a in branches:
            u, v, c = vout(a), T, k - 1
            forward_edges.append((u, v, c, len(dinic.graph[u])))
            dinic.add_edge(u, v, c)
        for v in range(n):
            u, vv, c = vin(v), vout(v), 1
            forward_edges.append((u, vv, c, len(dinic.graph[u])))
            dinic.add_edge(u, vv, c)
        for u, v in edges:
            uu, vv, c = vout(u), vin(v), INF
            forward_edges.append((uu, vv, c, len(dinic.graph[uu])))
            dinic.add_edge(uu, vv, c)
            uu, vv, c = vout(v), vin(u), INF
            forward_edges.append((uu, vv, c, len(dinic.graph[uu])))
            dinic.add_edge(uu, vv, c)

        flow = dinic.max_flow(S, T)

        if flow < k * (k - 1) // 2:
            return flow, None

        # Extract paths: for each forward edge, flow = cap - residual
        # residual is stored in dinic.graph[u][pos][1]
        # Build a directed multigraph of flow-carrying edges (only flow>0)
        # Then decompose into S->T paths.

        # Map node id -> original vertex or S/T
        # vin(v) -> v, vout(v) -> v, S -> None, T -> None

        # For path extraction, we want paths from S to T in terms of original vertices.
        # Each S->T path in the flow network goes: S -> vin(a) -> vout(a) -> vin(v1) -> vout(v1) -> ... -> vin(b) -> vout(b) -> T
        # The interior vertices are those v where vin(v)->vout(v) carries flow.

        # Build adjacency of flow edges: for each forward edge with flow>0, add directed edge u->v with flow amount.
        flow_adj = {}
        for (u, v, c, pos) in forward_edges:
            residual = dinic.graph[u][pos][1]
            f = c - residual
            if f > 0:
                for _ in range(f):
                    flow_adj.setdefault(u, []).append(v)

        # Now find k*(k-1)/2 paths from S to T in flow_adj (each edge used once since unit flows).
        # Use simple DFS path-finding repeatedly.
        paths_found = []
        needed = k * (k - 1) // 2
        # For efficiency, do DFS with backtracking
        def find_one_path():
            parent = {S: None}
            stack = [S]
            found = False
            while stack and not found:
                u = stack[-1]
            # Use recursive-style DFS
            def dfs(u):
                if u == T:
                    return True
                for v in flow_adj.get(u, []):
                    if v not in parent:
                        parent[v] = u
                        if dfs(v):
                            return True
                return False
            parent = {S: None}
            if not dfs(S):
                return None
            # Reconstruct
            path = []
            cur = T
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path

        # Remove edges as we use them (each unit of flow is one edge-use)
        for _ in range(needed):
            p = find_one_path()
            if p is None:
                break
            # Remove one unit of flow along this path
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                # Remove v from flow_adj[u]
                flow_adj[u].remove(v)
            paths_found.append(p)

        if len(paths_found) < needed:
            return flow, None

        # Convert each flow-network path to original-vertex path
        # Path: S, vin(a), vout(a), vin(v1), vout(v1), ..., vin(b), vout(b), T
        # Original: [a, v1, ..., b]
        result_paths = []
        for p in paths_found:
            # p[0] should be S, p[-1] should be T
            # Interior nodes alternate vin/vout
            orig = []
            for node in p[1:-1]:
                if node == S or node == T:
                    continue
                # vin(v) = 1+v, vout(v) = 1+n+v
                if node < 1 + n:
                    v = node - 1
                    orig.append(v)
                else:
                    v = node - 1 - n
                    # Only add if not already added (vin already added it)
                    if not orig or orig[-1] != v:
                        orig.append(v)
            result_paths.append(orig)

        return flow, result_paths

    # ---------- Max clique ----------
    def find_max_clique():
        best = []
        candidates = list(range(n))
        candidates.sort(key=lambda x: -len(adj[x]))

        def expand(clique, cand):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not cand:
                return
            for i, v in enumerate(cand):
                new_clique = clique + [v]
                new_cand = [u for u in cand[i + 1:] if u in adj[v]]
                expand(new_clique, new_cand)

        expand([], candidates)
        return best

    clique = find_max_clique()

    best_branches = None
    best_paths = None
    best_count = 0

    def update_best(branches, paths):
        nonlocal best_branches, best_paths, best_count
        if paths is not None and len(branches) > best_count:
            best_branches = list(branches)
            best_paths = list(paths)
            best_count = len(branches)

    # Try the clique with the flow-based extraction
    if len(clique) >= 2 and not time_up():
        flow, paths = build_and_extract(clique)
        if paths is not None:
            update_best(clique, paths)

    # ---------- Monotone k-search: try to find branch sets of increasing size ----------
    # For each k from best_count+1 upward, try random subsets of size k.
    # If flow >= k*(k-1)/2 and extraction succeeds, update best.

    if not time_up():
        for k in range(best_count + 1, min(n, 15) + 1):
            if time_up():
                break
            found = False
            # Try many random subsets of size k
            for attempt in range(500):
                if time_up():
                    break
                cand = random.sample(range(n), k)
                # Quick feasibility: check flow value only (faster than full extraction)
                flow, _ = build_and_extract(cand)
                if flow >= k * (k - 1) // 2:
                    # Re-extract (build_and_extract already does this, but we discarded paths)
                    flow2, paths = build_and_extract(cand)
                    if paths is not None:
                        update_best(cand, paths)
                        found = True
                        break
            if found:
                continue
            else:
                # If we can't find any subset of size k, maybe try k+1 with more attempts
                # but likely k is near the max. Break to save time.
                break

    # ---------- Local improvement: swap one branch for another ----------
    if best_branches is not None and not time_up():
        improved = True
        while improved and not time_up():
            improved = False
            cur_b = list(best_branches)
            cur_set = set(cur_b)
            for idx in range(len(cur_b)):
                if time_up():
                    break
                for v in range(n):
                    if v in cur_set:
                        continue
                    new_b = cur_b[:idx] + cur_b[idx + 1:] + [v]
                    new_b.sort()
                    flow, paths = build_and_extract(new_b)
                    if paths is not None and len(new_b) > best_count:
                        update_best(new_b, paths)
                        improved = True
                        break
                if improved:
                    break

    # ---------- Greedy expansion from best ----------
    if best_branches is not None and not time_up():
        current = list(best_branches)
        changed = True
        while changed and not time_up():
            changed = False
            for v in range(n):
                if v in current:
                    continue
                trial = current + [v]
                flow, paths = build_and_extract(trial)
                if paths is not None:
                    current = trial
                    changed = True
                    update_best(current, paths)
                    break
            if time_up():
                break

    # ---------- Random restarts ----------
    if not time_up():
        for restart in range(200):
            if time_up():
                break
            seed_size = min(n, max(2, best_count + random.randint(-1, 3)))
            seed = random.sample(range(n), seed_size)
            current = list(seed)
            flow, paths = build_and_extract(current)
            if paths is not None:
                update_best(current, paths)
            changed = True
            while changed and not time_up():
                changed = False
                for v in range(n):
                    if v in current:
                        continue
                    trial = current + [v]
                    flow, paths = build_and_extract(trial)
                    if paths is not None:
                        current = trial
                        changed = True
                        update_best(current, paths)
                        break
                if time_up():
                    break

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]

    return {'branches': branches_out, 'paths': paths_out}
