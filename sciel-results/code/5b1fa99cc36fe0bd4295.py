"""Clique subdivision via max-flow (vertex-disjoint multi-commodity) oracle."""
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 82.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # ---------- Dinic max-flow on an adjacency-list graph ----------
    class Dinic:
        def __init__(self, N):
            self.N = N
            self.adj = [[] for _ in range(N)]
            self.cap = []
            self.to = []

        def add_edge(self, u, v, c):
            self.adj[u].append(len(self.cap))
            self.cap.append(c)
            self.to.append(v)
            self.adj[v].append(len(self.cap))
            self.cap.append(0)
            self.to.append(u)

        def max_flow(self, s, t):
            N = self.N
            flow = 0
            INF = 10 ** 9
            while True:
                level = [-1] * N
                level[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    for ei in self.adj[u]:
                        if self.cap[ei] > 0:
                            v = self.to[ei]
                            if level[v] < 0:
                                level[v] = level[u] + 1
                                q.append(v)
                if level[t] < 0:
                    break
                it = [0] * N

                def dfs(u, f):
                    if u == t:
                        return f
                    for _ in range(it[u], len(self.adj[u])):
                        it[u] += 1
                        ei = self.adj[u][it[u] - 1]
                        if self.cap[ei] > 0:
                            v = self.to[ei]
                            if level[v] == level[u] + 1:
                                ret = dfs(v, min(f, self.cap[ei]))
                                if ret > 0:
                                    self.cap[ei] -= ret
                                    self.cap[ei ^ 1] += ret
                                    return ret
                    return 0

                while True:
                    pushed = dfs(s, INF)
                    if pushed == 0:
                        break
                    flow += pushed
            return flow

    # ---------- Build the vertex-split network for a branch set ----------
    # node id: 0=source, 1=sink, 2..2n-1 = split(v) for v in 0..n-1
    def build_network(branches, interior):
        # interior: set of non-branch vertices usable as path interiors
        N = 2 + 2 * n
        din = Dinic(N)
        src = 0
        sink = 1
        for v in range(n):
            din.add_edge(2 + 2 * v, 2 + 2 * v + 1, 1)  # vertex capacity
        for u, v in edges:
            din.add_edge(2 + 2 * u + 1, 2 + 2 * v, 40)
            din.add_edge(2 + 2 * v + 1, 2 + 2 * u, 40)
        for b in branches:
            din.add_edge(src, 2 + 2 * b, 100)
            din.add_edge(2 + 2 * b + 1, sink, 100)
        return din, src, sink

    # ---------- Extract a concrete path between two branches ----------
    def extract_path(din, a, b):
        # Walk from a's in-node following residual cap>0 edges toward b's in-node.
        # The flow network routes a->...->b. Reconstruct by BFS on residual graph
        # from a's out-node to b's in-node using edges with residual cap>0 that
        # were part of the flow (i.e. forward edges whose reverse gained cap).
        # Simpler: BFS on edges with cap>0 in forward direction from a_out to b_in.
        start = 2 + 2 * a + 1
        goal = 2 + 2 * b
        parent = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            if u == goal:
                break
            for ei in din.adj[u]:
                if din.cap[ei] > 0:
                    v = din.to[ei]
                    if v not in parent:
                        parent[v] = (u, ei)
                        q.append(v)
        if goal not in parent:
            return None
        # Reconstruct node sequence (split nodes), map back to original vertices.
        seq = [goal]
        cur = goal
        while parent[cur] is not None:
            u, ei = parent[cur]
            seq.append(u)
            cur = u
        seq.reverse()  # now from start to goal
        # Convert split node ids to original vertices:
        # in-node 2+2v -> v ; out-node 2+2v+1 -> v
        verts = []
        for node in seq:
            if node == 0 or node == 1:
                continue
            v = (node - 2) // 2
            if not verts or verts[-1] != v:
                verts.append(v)
        if verts[0] != a or verts[-1] != b:
            return None
        return verts

    # ---------- Oracle: can branch set be realized? returns paths or None ----------
    def solve_branch_set(branches):
        m = len(branches)
        if m < 2:
            return None
        need = m * (m - 1) // 2
        interior = set(range(n)) - set(branches)
        din, src, sink = build_network(branches, interior)
        flow = din.max_flow(src, sink)
        if flow < need:
            return None
        # For each unordered pair, extract a path.
        # We need distinct paths; the flow is vertex-disjoint so each pair's
        # unit flow is on its own interior. Extract per pair.
        paths = {}
        # Determine which pairs are "connected" by finding, for each branch a,
        # the flow reaching it. We do a per-pair residual BFS (cheap, n small).
        for i in range(m):
            for j in range(i + 1, m):
                p = extract_path(din, branches[i], branches[j])
                if p is None:
                    return None
                paths[(i, j)] = p
        # Verify interior disjointness and validity (defensive).
        used = set()
        for (i, j), p in paths.items():
            interior_set = set(p[1:-1])
            if interior_set & used:
                return None
            if interior_set & set(branches):
                return None
            used |= interior_set
            # adjacency check
            for x, y in zip(p, p[1:]):
                if y not in adj[x]:
                    return None
        out = [paths[(i, j)] for i in range(m) for j in range(i + 1, m)]
        return out

    # ---------- Search over branch sets ----------
    best_branches = None
    best_paths = None
    best_count = 0

    def update_best(branches, paths):
        nonlocal best_branches, best_paths, best_count
        if paths is not None and len(branches) > best_count:
            best_branches = list(branches)
            best_paths = paths
            best_count = len(branches)

    # Initial: max clique (a clique is always realizable via direct edges).
    def find_max_clique():
        best = []
        cands = sorted(range(n), key=lambda x: -len(adj[x]))

        def expand(clique, cand):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not cand:
                return
            for i, v in enumerate(cand):
                new_cand = [u for u in cand[i + 1:] if u in adj[v]]
                expand(clique + [v], new_cand)

        expand([], cands)
        return best

    clique = find_max_clique()
    if len(clique) >= 2:
        update_best(clique, solve_branch_set(clique))

    # Greedy expansion using the flow oracle (order-independent).
    current = list(clique)
    changed = True
    while changed and not time_up():
        changed = False
        for v in range(n):
            if v in current:
                continue
            trial = current + [v]
            paths = solve_branch_set(trial)
            if paths is not None:
                current = trial
                update_best(current, paths)
                changed = True
                break
        if time_up():
            break

    # Random restarts: seed with a random clique-ish set, then flow-greedy expand.
    if not time_up():
        for restart in range(300):
            if time_up():
                break
            # Build a random seed: start from a random vertex, add random neighbors
            # to get a dense seed, then expand.
            seed = [random.randrange(n)]
            for _ in range(random.randint(1, 4)):
                if not seed:
                    break
                base = random.choice(seed)
                nbrs = list(adj[base])
                if nbrs:
                    cand = random.choice(nbrs)
                    if cand not in seed:
                        seed.append(cand)
            # Optionally include a couple of random extra vertices.
            for _ in range(random.randint(0, 2)):
                cand = random.randrange(n)
                if cand not in seed:
                    seed.append(cand)
            current = list(seed)
            changed = True
            while changed and not time_up():
                changed = False
                for v in range(n):
                    if v in current:
                        continue
                    trial = current + [v]
                    paths = solve_branch_set(trial)
                    if paths is not None:
                        current = trial
                        update_best(current, paths)
                        changed = True
                        break
                if time_up():
                    break

    # Fallback.
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]
    return {'branches': branches_out, 'paths': paths_out}
