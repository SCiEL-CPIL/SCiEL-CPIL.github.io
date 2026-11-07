"""Maximum clique subdivision in a pinned graph.

Structural change vs. seed:
  - Outer: branch-and-bound over vertex subsets maximizing |branches|.
  - Inner: for a candidate branch set B of size m, build a vertex-split
    network with ONE unit of flow per unordered pair (a,b) in B.
    Non-branch vertices have capacity 1; branch vertices are split with
    infinite capacity but each branch's in/out copy is capped at (m-1)
    total usage (each branch touches m-1 paths).  A super-source feeds
    each branch's in-copy with capacity (m-1) and each branch's out-copy
    feeds a super-sink with capacity (m-1).  Edge arcs have capacity 1.
    If max flow == m*(m-1)//2, the flow decomposition yields a set of
    vertex-disjoint (interior) simple paths, one per pair.
  - Path extraction: each unit of flow from a branch to another branch
    is a path; we extract them from the residual flow on the split graph.
"""
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t0 = time.time()
    DEADLINE = 82.0  # safety margin under 90s

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t0) > DEADLINE

    # ---------- Dinic with parent tracking for flow extraction ----------
    class Dinic:
        def __init__(self, N):
            self.N = N
            self.g = [[] for _ in range(N)]  # each: [to, cap, rev_idx]

        def add_edge(self, u, v, c):
            self.g[u].append([v, c, len(self.g[v])])
            self.g[v].append([u, 0, len(self.g[u]) - 1])

        def bfs(self, s, t):
            self.level = [-1] * self.N
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for e in self.g[u]:
                    v, c, _ = e
                    if c > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[t] >= 0

        def dfs(self, u, t, f):
            if u == t:
                return f
            gu = self.g[u]
            while self.it[u] < len(gu):
                e = gu[self.it[u]]
                v, c, rev = e
                if c > 0 and self.level[v] == self.level[u] + 1:
                    d = self.dfs(v, t, min(f, c))
                    if d > 0:
                        e[1] -= d
                        self.g[v][rev][1] += d
                        return d
                self.it[u] += 1
            return 0

        def max_flow(self, s, t):
            flow = 0
            INF = 10 ** 9
            while self.bfs(s, t):
                self.it = [0] * self.N
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    # ---------- Build the vertex-split flow network for branch set B ----------
    # Node ids:
    #   S = 0, T = 1
    #   for each vertex v: in_v = 2 + v, out_v = 2 + n + v
    #   Total nodes = 2 + 2n
    # Arcs:
    #   S -> in_a  (cap m-1) for each branch a
    #   out_a -> T  (cap m-1) for each branch a
    #   in_v -> out_v : cap INF if v in B else 1
    #   for each undirected edge (u,v): out_u -> in_v cap 1, out_v -> in_u cap 1
    # Required total flow: m*(m-1)//2
    #
    # After max flow, extract paths: each unit of flow that left S via in_a
    # and entered T via out_b corresponds to a path from a to b.
    # We trace each unit by walking the residual: for each branch a, look at
    # outgoing flow from in_a (i.e., edges in_a->out_a and in_a->... actually
    # flow on S->in_a tells how many units entered a).  Simpler: after flow,
    # for each original edge arc, the flow is the residual reverse capacity.
    # We build a directed multigraph of unit flows and decompose.

    def solve_paths(branches):
        """Return list of paths (each a list of original vertex ids) or None."""
        m = len(branches)
        if m < 2:
            return None
        required = m * (m - 1) // 2
        branch_set = set(branches)
        S = 0
        T = 1
        N = 2 + 2 * n
        dinic = Dinic(N)

        def in_node(v):
            return 2 + v

        def out_node(v):
            return 2 + n + v

        # S -> in_branch, out_branch -> T
        for a in branches:
            dinic.add_edge(S, in_node(a), m - 1)
            dinic.add_edge(out_node(a), T, m - 1)

        # vertex capacity
        INF = 10 ** 9
        for v in range(n):
            cap = INF if v in branch_set else 1
            dinic.add_edge(in_node(v), out_node(v), cap)

        # edge arcs (directed both ways, cap 1 each)
        for u, v in edges:
            dinic.add_edge(out_node(u), in_node(v), 1)
            dinic.add_edge(out_node(v), in_node(u), 1)

        flow = dinic.max_flow(S, T)
        if flow < required:
            return None

        # ---- Extract paths from the flow ----
        # For each directed arc in the residual, flow = reverse cap.
        # Build a list of unit-flow edges: (from_node, to_node) with multiplicity.
        # We'll represent flow as a dict (u,v) -> count.
        flow_edges = {}
        for u in range(N):
            gu = dinic.g[u]
            for e in gu:
                v, cap, rev = e
                # forward residual cap is cap; original capacity was cap + reverse_cap
                rev_e = dinic.g[v][rev]
                orig_cap = cap + rev_e[1]
                if orig_cap == 0:
                    continue
                f = rev_e[1]  # flow = reverse residual capacity
                if f > 0:
                    key = (u, v)
                    flow_edges[key] = flow_edges.get(key, 0) + f

        # Now decompose into S->T paths in the flow graph.
        # Each path goes S -> in_a -> ... -> out_b -> T.
        # We'll do a standard flow decomposition: repeatedly find an S-T path
        # in the flow graph (edges with positive flow), remove one unit.
        # Build adjacency of flow graph.
        from collections import defaultdict
        f_adj = defaultdict(lambda: defaultdict(int))
        for (u, v), c in flow_edges.items():
            f_adj[u][v] += c

        paths = []
        # We need exactly 'required' paths.  Each path starts at S and ends at T.
        # Trace: S -> in_a (a is branch) -> out_a -> ... -> out_b -> T
        # The "interior" of the path in original vertices:
        #   in_a -> out_a is the branch vertex a itself (no edge in original graph, it's the vertex-capacity arc)
        #   Then out_a -> in_x means edge (a, x) in original graph
        #   in_x -> out_x is vertex x
        #   out_x -> in_y means edge (x, y)
        #   ...
        #   out_b -> T
        # So the original path is: a, x1, x2, ..., b
        # where the sequence of "out -> in" arcs gives consecutive original vertices.

        for _ in range(required):
            if time_up():
                return None
            # BFS in flow graph from S to T
            parent = {S: None}
            q = deque([S])
            found = False
            while q:
                u = q.popleft()
                if u == T:
                    found = True
                    break
                for v, c in f_adj[u].items():
                    if c > 0 and v not in parent:
                        parent[v] = u
                        q.append(v)
            if not found:
                return None  # shouldn't happen if flow == required
            # Reconstruct path
            path_nodes = []
            cur = T
            while cur is not None:
                path_nodes.append(cur)
                cur = parent[cur]
            path_nodes.reverse()
            # path_nodes: [S, in_a, out_a, in_x1, out_x1, ..., out_b, T]
            # Convert to original vertices
            orig = []
            i = 1  # skip S
            while i < len(path_nodes) - 1:  # skip T
                node = path_nodes[i]
                # node is either in_v or out_v
                if node < 2 + n:
                    v = node - 2
                    orig.append(v)
                    i += 2  # skip in_v and out_v
                else:
                    # out_v: this is the "out" part of a vertex we already added? 
                    # Actually the pattern is in_a, out_a, in_x, out_x, ...
                    # So we should only add when we see in_v.
                    i += 1
            # Verify: orig should be a path from a to b
            # Remove 1 unit of flow along this path
            for idx in range(len(path_nodes) - 1):
                u, v = path_nodes[idx], path_nodes[idx + 1]
                f_adj[u][v] -= 1
                if f_adj[u][v] <= 0:
                    del f_adj[u][v]
            if not f_adj[u]:
                del f_adj[u]
            paths.append(orig)

        # Verify each path
        for p in paths:
            if len(p) < 2:
                return None
            a, b = p[0], p[-1]
            if a not in branch_set or b not in branch_set:
                return None
            if a == b:
                return None
            # Check consecutive adjacency
            for i in range(len(p) - 1):
                if p[i + 1] not in adj[p[i]]:
                    return None
            # Check no interior vertex is a branch
            for x in p[1:-1]:
                if x in branch_set:
                    return None
            # Check no repeated vertices (simple path)
            if len(set(p)) != len(p):
                return None

        # Check all pairs covered exactly once
        pair_count = {}
        for p in paths:
            key = tuple(sorted([p[0], p[-1]]))
            pair_count[key] = pair_count.get(key, 0) + 1
        for a, b in combinations(branches, 2):
            key = tuple(sorted([a, b]))
            if pair_count.get(key, 0) != 1:
                return None
        if len(paths) != required:
            return None

        # Check interior vertex-disjointness
        used_interior = set()
        for p in paths:
            interior = set(p[1:-1])
            if interior & used_interior:
                return None
            used_interior.update(interior)

        return paths

    # ---------- Branch-and-bound over vertex subsets ----------
    # We want to maximize |B| such that solve_paths(B) is not None.
    # Strategy: try sizes from n down to 2.  For each size k, enumerate
    # subsets (with pruning) and test feasibility.
    # To make this tractable, use a greedy + local search hybrid, but the
    # feasibility oracle is now exact (max-flow), so we can be more aggressive.

    # Precompute degree ordering
    deg_order = sorted(range(n), key=lambda x: -len(adj[x]))

    best_branches = None
    best_paths = None
    best_count = 0

    def update(b, p):
        nonlocal best_branches, best_paths, best_count
        if p is not None and len(b) > best_count:
            best_branches = list(b)
            best_paths = list(p)
            best_count = len(b)

    # Try the full graph, then reduce
    # First: greedy construction using the exact oracle
    current = []
    for v in deg_order:
        if time_up():
            break
        trial = current + [v]
        if len(trial) < 2:
            current = trial
            continue
        p = solve_paths(trial)
        if p is not None:
            current = trial
            update(current, p)
    # This greedy might get stuck; try multiple orderings
    for restart in range(50):
        if time_up():
            break
        order = list(deg_order)
        random.shuffle(order)
        # Bias toward high-degree
        order.sort(key=lambda x: -len(adj[x]) + random.random() * 4)
        current = []
        for v in order:
            if time_up():
                break
            trial = current + [v]
            if len(trial) < 2:
                current = trial
                continue
            p = solve_paths(trial)
            if p is not None:
                current = trial
                update(current, p)

    # Local improvement: try to add a vertex, or swap
    improved = True
    while improved and not time_up():
        improved = False
        cur = list(best_branches) if best_branches else []
        cur_set = set(cur)
        # Try adding each non-branch vertex
        for v in deg_order:
            if time_up():
                break
            if v in cur_set:
                continue
            trial = cur + [v]
            p = solve_paths(trial)
            if p is not None:
                update(trial, p)
                improved = True
                cur = list(best_branches)
                cur_set = set(cur)
                break
        # Try 1-swap: remove one, add one
        if not improved and len(cur) >= 2:
            for i in range(len(cur)):
                if time_up():
                    break
                for v in deg_order:
                    if time_up():
                        break
                    if v in cur_set:
                        continue
                    trial = cur[:i] + cur[i + 1:] + [v]
                    p = solve_paths(trial)
                    if p is not None and len(trial) > best_count:
                        update(trial, p)
                        improved = True
                        break
                if improved:
                    break

    # Try to push for larger sizes via random subset sampling with exact oracle
    target = best_count + 1
    while target <= n and not time_up():
        found = False
        attempts = 0
        while attempts < 500 and not time_up():
            attempts += 1
            cand = random.sample(range(n), target)
            p = solve_paths(cand)
            if p is not None:
                update(cand, p)
                found = True
                break
        if found:
            target += 1
        else:
            break

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]

    return {'branches': branches_out, 'paths': paths_out}
