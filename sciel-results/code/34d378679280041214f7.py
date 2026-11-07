"""Search for maximum branch count in a pinned graph clique subdivision.

Strategy:
  - For a candidate branch set B, test feasibility via max-flow:
      * split every non-branch vertex v into v_in -> v_out with capacity 1
      * branch vertices get infinite capacity
      * add a super-source S and super-sink T
      * for each unordered pair (a,b) of branches, add an edge a -> b (and b -> a)
        with capacity 1, and connect S->a and b->T appropriately... 
      Actually simpler: we need internally-disjoint paths for every pair.
      Model: source S connects to each branch a (cap 1 per (a,b) pair is tricky).
      
  Better model for disjoint paths per pair:
      We need |B| choose 2 edge-disjoint (interior-disjoint) paths.
      Standard trick: create a super-source S and super-sink T.
      For each branch a, add edge S -> a with capacity (|B|-1) [a is endpoint of |B|-1 paths]
      For each branch a, add edge a -> T with capacity (|B|-1)
      For each interior vertex v, split v_in -> v_out cap 1.
      For each original edge (u,v):
         - if both u,v are branches: add u->v cap 1 AND v->u cap 1 (direct edge path)
         - if u branch, v interior: add u -> v_in cap 1
         - if v branch, u interior: add u_out -> v cap 1
         - if both interior: add u_out -> v_in cap 1
      Then max flow from S to T should equal |B|(|B|-1)/2 * 2? 
      Hmm, each path a->b uses 1 unit from S->a and 1 unit from b->T.
      Total flow needed = number of paths = |B|(|B|-1)/2.
      But S->a has capacity |B|-1 (a participates in |B|-1 paths as start).
      And a->T has capacity |B|-1 (a participates in |B|-1 paths as end).
      Total capacity out of S = |B|(|B|-1), total into T = |B|(|B|-1).
      Each path contributes 1 to flow. We need exactly |B|(|B|-1)/2 paths.
      So max flow >= |B|(|B|-1)/2 means feasible.
      
      Wait, but a path from a to b: S->a (1), then interior, then b->T (1). Flow = 1 per path.
      Yes. So we need max_flow >= C(|B|,2).
      
  - Use this as a fast feasibility test (Dinic's is fast for small graphs).
  - Search: start from a clique, greedily add vertices, use max-flow to verify.
  - Also use min-cut to identify which branch to drop when infeasible.
"""
import time
import random
import sys
from collections import deque

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

    # Dinic's max flow
    class Dinic:
        def __init__(self, N):
            self.N = N
            self.graph = [[] for _ in range(N)]
            self.cap = []

        def add_edge(self, u, v, c):
            # forward edge index = len(self.graph[u])
            self.graph[u].append([v, c, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])
            self.cap.append(c)

        def bfs(self, s, t):
            self.level = [-1] * self.N
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v, c, _ in self.graph[u]:
                    if c > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[t] >= 0

        def dfs(self, u, t, f):
            if u == t:
                return f
            while self.it[u] < len(self.graph[u]):
                v, c, rev = self.graph[u][self.it[u]]
                if c > 0 and self.level[v] == self.level[u] + 1:
                    d = self.dfs(v, t, min(f, c))
                    if d > 0:
                        self.graph[u][self.it[u]][1] -= d
                        self.graph[v][rev][1] += d
                        return d
                self.it[u] += 1
            return 0

        def max_flow(self, s, t):
            flow = 0
            INF = float('inf')
            while self.bfs(s, t):
                self.it = [0] * self.N
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    # Build flow network for a given branch set
    # Returns (max_flow_value, feasible)
    def build_flow_network(branches):
        """
        branches: list of branch vertex indices
        Returns max flow value. Feasible if max_flow >= len(branches)*(len(branches)-1)//2
        """
        m = len(branches)
        if m < 2:
            return 0
        branch_set = set(branches)
        # Node mapping:
        # 0: super source S
        # 1: super sink T
        # For each vertex v:
        #   if v is branch: node_in(v) = 2 + v, node_out(v) = 2 + n + v
        #   if v is interior: node_in(v) = 2 + v, node_out(v) = 2 + n + v
        # Total nodes: 2 + 2*n
        S = 0
        T = 1
        def node_in(v):
            return 2 + v
        def node_out(v):
            return 2 + n + v
        N_nodes = 2 + 2 * n
        dinic = Dinic(N_nodes)

        # S -> branch (capacity m-1)
        for a in branches:
            dinic.add_edge(S, node_in(a), m - 1)
        # branch -> T (capacity m-1)
        for a in branches:
            dinic.add_edge(node_out(a), T, m - 1)

        # Interior vertices: v_in -> v_out cap 1
        for v in range(n):
            if v in branch_set:
                # Branch: v_in -> v_out with infinite capacity
                dinic.add_edge(node_in(v), node_out(v), float('inf'))
            else:
                dinic.add_edge(node_in(v), node_out(v), 1)

        # Edges
        for u, v in edges:
            if u in branch_set and v in branch_set:
                # Direct edge between branches
                dinic.add_edge(node_out(u), node_in(v), 1)
                dinic.add_edge(node_out(v), node_in(u), 1)
            elif u in branch_set:
                # u is branch, v is interior
                dinic.add_edge(node_out(u), node_in(v), 1)
                dinic.add_edge(node_out(v), node_in(u), 1)
            elif v in branch_set:
                # v is branch, u is interior
                dinic.add_edge(node_out(u), node_in(v), 1)
                dinic.add_edge(node_out(v), node_in(u), 1)
            else:
                # Both interior
                dinic.add_edge(node_out(u), node_in(v), 1)
                dinic.add_edge(node_out(v), node_in(u), 1)

        return dinic.max_flow(S, T)

    def is_feasible(branches):
        m = len(branches)
        if m < 2:
            return True
        required = m * (m - 1) // 2
        flow = build_flow_network(branches)
        return flow >= required

    # Find actual paths given a feasible branch set using BFS with used interiors
    def find_paths(branches):
        m = len(branches)
        if m < 2:
            return None
        used_interior = set()
        paths = {}
        pairs = []
        for i in range(m):
            for j in range(i + 1, m):
                pairs.append((i, j))
        # Sort: direct edges first
        def pair_key(p):
            i, j = p
            a, b = branches[i], branches[j]
            direct = 0 if b in adj[a] else 1
            return (direct, -len(adj[a]), -len(adj[b]))
        pairs.sort(key=pair_key)

        def bfs_path(s, t, banned):
            if s == t:
                return None
            if t in adj[s] and t not in banned:
                return [s, t]
            parent = {s: None}
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v == s:
                        continue
                    if v in parent:
                        continue
                    if v == t:
                        path = [v]
                        cur = u
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        path.reverse()
                        return path
                    if v in banned:
                        continue
                    parent[v] = u
                    q.append(v)
            return None

        for (i, j) in pairs:
            a, b = branches[i], branches[j]
            banned = set(used_interior)
            banned.update(branches)
            banned.discard(a)
            banned.discard(b)
            path = bfs_path(a, b, banned)
            if path is None:
                # Try different orderings - retry with different pair order
                return None
            interior = set(path[1:-1])
            if interior & used_interior:
                return None
            used_interior.update(interior)
            paths[(i, j)] = path

        result_paths = []
        for (i, j) in pairs:
            result_paths.append(paths[(i, j)])
        return result_paths

    # Find max clique
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

    # Try the clique
    if len(clique) >= 2:
        if is_feasible(clique):
            paths = find_paths(clique)
            if paths is not None:
                update_best(clique, paths)

    # Greedy expansion
    current = list(clique)
    changed = True
    while changed and not time_up():
        changed = False
        for v in range(n):
            if v in current:
                continue
            trial = current + [v]
            if is_feasible(trial):
                paths = find_paths(trial)
                if paths is not None:
                    current = trial
                    changed = True
                    update_best(current, paths)
                    break
        if time_up():
            break

    # Random restarts
    if not time_up():
        for restart in range(200):
            if time_up():
                break
            seed_size = min(n, max(2, best_count + random.randint(-1, 3)))
            seed = random.sample(range(n), seed_size)
            current = list(seed)
            if is_feasible(current):
                paths = find_paths(current)
                if paths is not None:
                    update_best(current, paths)
            changed = True
            while changed and not time_up():
                changed = False
                for v in range(n):
                    if v in current:
                        continue
                    trial = current + [v]
                    if is_feasible(trial):
                        paths = find_paths(trial)
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
