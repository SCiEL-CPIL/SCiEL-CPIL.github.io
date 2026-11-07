"""Search for maximum branch count in a pinned graph clique subdivision.

Combined strategy:
  - Max-flow feasibility test (Dinic's) for candidate branch sets.
  - Greedy expansion from max clique.
  - Random restarts with multiple BFS orderings.
  - Local improvement: swap one branch for another.
  - Backtracking search for path packing with multiple orderings.
"""
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

    # Dinic's max flow
    class Dinic:
        def __init__(self, N):
            self.N = N
            self.graph = [[] for _ in range(N)]
            self.cap = []

        def add_edge(self, u, v, c):
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

    def build_flow_network(branches):
        m = len(branches)
        if m < 2:
            return 0
        branch_set = set(branches)
        S = 0
        T = 1

        def node_in(v):
            return 2 + v

        def node_out(v):
            return 2 + n + v

        N_nodes = 2 + 2 * n
        dinic = Dinic(N_nodes)

        for a in branches:
            dinic.add_edge(S, node_in(a), m - 1)
        for a in branches:
            dinic.add_edge(node_out(a), T, m - 1)

        for v in range(n):
            if v in branch_set:
                dinic.add_edge(node_in(v), node_out(v), float('inf'))
            else:
                dinic.add_edge(node_in(v), node_out(v), 1)

        for u, v in edges:
            if u in branch_set and v in branch_set:
                dinic.add_edge(node_out(u), node_in(v), 1)
                dinic.add_edge(node_out(v), node_in(u), 1)
            elif u in branch_set:
                dinic.add_edge(node_out(u), node_in(v), 1)
                dinic.add_edge(node_out(v), node_in(u), 1)
            elif v in branch_set:
                dinic.add_edge(node_out(u), node_in(v), 1)
                dinic.add_edge(node_out(v), node_in(u), 1)
            else:
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

    def bfs_path(s, t, banned):
        if s == t:
            return None
        if t in adj[s]:
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

    def find_paths(branches, order=None):
        m = len(branches)
        if m < 2:
            return None
        used_interior = set()
        paths = {}
        if order is None:
            pairs = []
            for i in range(m):
                for j in range(i + 1, m):
                    pairs.append((i, j))

            def pair_key(p):
                i, j = p
                a, b = branches[i], branches[j]
                direct = 0 if b in adj[a] else 1
                return (direct, -len(adj[a]), -len(adj[b]))
            pairs.sort(key=pair_key)
        else:
            pairs = order

        for (i, j) in pairs:
            a, b = branches[i], branches[j]
            banned = set(used_interior)
            banned.update(branches)
            banned.discard(a)
            banned.discard(b)
            path = bfs_path(a, b, banned)
            if path is None:
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

    def find_paths_multi_order(branches, num_orders=5):
        m = len(branches)
        if m < 2:
            return None
        best = None
        # Try default order
        p = find_paths(branches)
        if p is not None:
            return p
        # Try random orders
        for _ in range(num_orders):
            pairs = []
            for i in range(m):
                for j in range(i + 1, m):
                    pairs.append((i, j))
            random.shuffle(pairs)
            p = find_paths(branches, pairs)
            if p is not None:
                return p
        return None

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
            paths = find_paths_multi_order(clique, 10)
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
                paths = find_paths_multi_order(trial, 5)
                if paths is not None:
                    current = trial
                    changed = True
                    update_best(current, paths)
                    break
        if time_up():
            break

    # Local improvement: try swapping one branch for another
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
                    if is_feasible(new_b):
                        paths = find_paths_multi_order(new_b, 5)
                        if paths is not None and len(new_b) > best_count:
                            update_best(new_b, paths)
                            improved = True
                            break
                if improved:
                    break

    # Random restarts
    if not time_up():
        for restart in range(300):
            if time_up():
                break
            seed_size = min(n, max(2, best_count + random.randint(-1, 3)))
            seed = random.sample(range(n), seed_size)
            current = list(seed)
            if is_feasible(current):
                paths = find_paths_multi_order(current, 5)
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
                        paths = find_paths_multi_order(trial, 5)
                        if paths is not None:
                            current = trial
                            changed = True
                            update_best(current, paths)
                            break
                if time_up():
                    break

    # Backtracking search for larger branch sets
    if not time_up():
        # Try to find branch sets of size best_count+1, best_count+2, etc.
        target = best_count + 1
        while target <= n and not time_up():
            found = False
            # Sample random subsets of size target
            for attempt in range(100):
                if time_up():
                    break
                if target > n:
                    break
                cand = random.sample(range(n), target)
                if is_feasible(cand):
                    paths = find_paths_multi_order(cand, 10)
                    if paths is not None:
                        update_best(cand, paths)
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
