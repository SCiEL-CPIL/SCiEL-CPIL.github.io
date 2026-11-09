import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin under 90s kill

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    degrees = [len(adj[i]) for i in range(n)]

    def time_ok():
        return time.time() < deadline

    # ---------------- Max-flow (Dinic) with vertex splitting ----------------
    # Necessity test: a set of branches admits a subdivision iff there exist
    # k(k-1)/2 vertex-disjoint paths (interiors disjoint, avoiding other branches).
    # In a simple undirected graph, Menger's theorem makes vertex-disjoint path
    # existence equivalent to a max-flow of value k(k-1)/2 in the vertex-split
    # network (interior vertices capacity 1, branch vertices capacity INF).
    def dinic_maxflow(num_nodes, edges_list, src, snk, target):
        """edges_list: list of (u, v, cap). Returns max flow (capped at target)."""
        # Build adjacency with edge objects
        to = []
        cap = []
        nxt = []
        head = [-1] * num_nodes
        def add_edge(u, v, c):
            idx = len(to)
            to.append(v); cap.append(c); nxt.append(head[u]); head[u] = idx
            to.append(u); cap.append(0); nxt.append(head[v]); head[v] = idx + 1
        for (u, v, c) in edges_list:
            add_edge(u, v, c)
        INF = 10**9
        flow = 0
        while flow < target and time_ok():
            # BFS level
            level = [-1] * num_nodes
            q = deque([src])
            level[src] = 0
            while q:
                u = q.popleft()
                e = head[u]
                while e != -1:
                    if cap[e] > 0 and level[to[e]] == -1:
                        level[to[e]] = level[u] + 1
                        q.append(to[e])
                    e = nxt[e]
            if level[snk] == -1:
                break
            # DFS blocking flow (iterative)
            it = [head[:] ]
            def dfs_iter():
                # stack-based
                stack = [(src, 0, INF, [])]
                while stack:
                    u, edge_idx, f, path_edges = stack.pop()
                    if u == snk:
                        # augment
                        mn = f
                        for ei in path_edges:
                            if cap[ei] < mn:
                                mn = cap[ei]
                        for ei in path_edges:
                            cap[ei] -= mn
                            cap[ei ^ 1] += mn
                        return mn
                    e = it[u]
                    advanced = False
                    while e != -1:
                        v = to[e]
                        if cap[e] > 0 and level[v] == level[u] + 1:
                            it[u] = e  # will advance after this edge is exhausted
                            stack.append((u, e, f, path_edges))  # restore later
                            stack.append((v, it[v], min(f, cap[e]), path_edges + [e]))
                            advanced = True
                            break
                        e = nxt[e]
                    if not advanced:
                        it[u] = -1
                return 0
            # Simpler: recursive DFS with sys.setrecursionlimit safe for 2n+2 ~ 50 nodes
            def dfs(u, f):
                if u == snk:
                    return f
                e = it[u]
                while e != -1:
                    v = to[e]
                    if cap[e] > 0 and level[v] == level[u] + 1:
                        pushed = dfs(v, min(f, cap[e]))
                        if pushed > 0:
                            cap[e] -= pushed
                            cap[e ^ 1] += pushed
                            return pushed
                    e = nxt[e]
                    it[u] = e
                return 0
            import sys
            old_limit = sys.getrecursionlimit()
            if old_limit < 5000:
                sys.setrecursionlimit(5000)
            while time_ok():
                pushed = dfs(src, INF)
                if pushed == 0:
                    break
                flow += pushed
                if flow >= target:
                    break
            sys.setrecursionlimit(old_limit)
        return flow

    def flow_feasible(branches):
        """Return True iff max-flow value reaches k(k-1)/2 (necessary condition)."""
        k = len(branches)
        if k < 2:
            return False
        target = k * (k - 1) // 2
        branches_set = set(branches)
        # Node indexing: v_in = 2*v, v_out = 2*v+1, SRC=2n, SNK=2n+1
        total_nodes = 2 * n + 2
        SRC = 2 * n
        SNK = 2 * n + 1
        INF = 10**6
        edges_list = []
        for v in range(n):
            vin, vout = 2 * v, 2 * v + 1
            if v in branches_set:
                edges_list.append((vin, vout, INF))
            else:
                edges_list.append((vin, vout, 1))
        for u in range(n):
            for v in adj_list[u]:
                if u < v:
                    uout, vin = 2 * u + 1, 2 * v
                    vout, uin = 2 * v + 1, 2 * u
                    edges_list.append((uout, vin, INF))
                    edges_list.append((vout, uin, INF))
        # Connect SRC to each branch's vin, each branch's vout to SNK
        for b in branches:
            edges_list.append((SRC, 2 * b, INF))
            edges_list.append((2 * b + 1, SNK, INF))
        val = dinic_maxflow(total_nodes, edges_list, SRC, SNK, target)
        return val >= target

    # ---------------- Path finding with backtracking ----------------
    def bfs_candidates(s, t, blocked, max_paths=12, max_len=12):
        """BFS to find up to max_paths short simple paths s->t avoiding blocked interiors."""
        if s == t:
            return [[s]]
        if t in adj[s]:
            return [[s, t]]
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        found = []
        while queue and len(found) < max_paths and time_ok():
            cur = queue.popleft()
            for nb in adj_list[cur]:
                if nb in blocked and nb != s and nb != t:
                    continue
                if nb in visited:
                    continue
                visited.add(nb)
                parent[nb] = cur
                if nb == t:
                    path = [t]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    if len(path) - 2 <= max_len:
                        found.append(path)
                    del parent[nb]
                    visited.discard(nb)
                else:
                    queue.append(nb)
        return found

    def solve_disjoint(branches, max_restarts=40):
        """Backtracking with candidate paths. Returns list of paths (one per pair) or None."""
        k = len(branches)
        if k < 2:
            return None
        branches_set = set(branches)
        pairs = [(branches[i], branches[j]) for i in range(k) for j in range(i + 1, k)]
        num_pairs = len(pairs)

        # Precompute a "constraint score" for ordering pairs: fewer common non-branch
        # neighbors => more constrained => assign earlier.
        def pair_score(p):
            u, v = p
            common = (adj[u] & adj[v]) - branches_set
            return len(common)

        base_order = sorted(pairs, key=pair_score)

        for restart in range(max_restarts):
            if not time_ok():
                return None
            if restart == 0:
                order = base_order
            else:
                order = list(base_order)
                random.shuffle(order)

            path_map = {}
            used_interior = set()

            def solve(idx):
                if idx == num_pairs:
                    return True
                u, v = order[idx]
                blocked = used_interior | (branches_set - {u, v})
                cands = bfs_candidates(u, v, blocked, max_paths=8, max_len=10)
                if not cands:
                    return False
                cands.sort(key=len)
                for path in cands:
                    interior = set(path[1:-1])
                    if interior & used_interior:
                        continue
                    if interior & (branches_set - {u, v}):
                        continue
                    path_map[(min(u, v), max(u, v))] = path
                    used_interior |= interior
                    if solve(idx + 1):
                        return True
                    used_interior -= interior
                    del path_map[(min(u, v), max(u, v))]
                return False

            if solve(0):
                output = []
                for i in range(k):
                    for j in range(i + 1, k):
                        a, b = branches[i], branches[j]
                        key = (min(a, b), max(a, b))
                        p = path_map[key]
                        if p[0] == branches[i]:
                            output.append(p)
                        else:
                            output.append(p[::-1])
                return output
        return None

    # ---------------- Maximum clique (Bron-Kerbosch with pivot) ----------------
    def find_max_clique():
        best = []
        def bk(R, Pset, X):
            nonlocal best
            if not time_ok():
                return
            if not Pset and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if Pset:
                pivot_candidates = list(Pset | X)
            else:
                pivot_candidates = list(X)
            if not pivot_candidates:
                return
            u = max(pivot_candidates, key=lambda v: len(adj[v] & Pset))
            candidates = [v for v in Pset if v not in adj[u]]
            candidates.sort(key=lambda x: -degrees[x])
            for v in candidates:
                if not time_ok():
                    return
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset = Pset - {v}
                X = X | {v}
        bk(frozenset(), frozenset(range(n)), frozenset())
        return best

    # ---------------- Main search ----------------
    best_result = None
    best_k = 0

    def try_accept(branches):
        nonlocal best_result, best_k
        if len(branches) <= best_k:
            return False
        # Flow feasibility check (necessary condition, fast filter)
        if not flow_feasible(branches):
            return False
        paths = solve_disjoint(branches, max_restarts=30)
        if paths is not None:
            best_result = (sorted(branches), paths)
            best_k = len(branches)
            return True
        return False

    # 1. Start from max clique
    max_clique = sorted(find_max_clique())
    if len(max_clique) >= 2:
        try_accept(max_clique)

    # 2. Try subsets of the clique (descending size) if full clique failed
    if best_k < len(max_clique):
        for size in range(len(max_clique) - 1, 1, -1):
            if not time_ok():
                break
            found = False
            for attempt in range(30):
                if not time_ok():
                    break
                subset = list(random.sample(max_clique, size))
                if try_accept(subset):
                    found = True
                    break
            if found:
                break
            # Exhaustive removal for small gaps
            if len(max_clique) - size <= 3:
                for remove_set in combinations(max_clique, len(max_clique) - size):
                    if not time_ok():
                        break
                    subset = [v for v in max_clique if v not in remove_set]
                    if try_accept(subset):
                        found = True
                        break
                if found:
                    break

    # 3. Grow: add one vertex at a time
    if best_result is not None:
        current_branches = list(best_result[0])
        improved = True
        while improved and time_ok():
            improved = False
            for v in range(n):
                if v in set(current_branches):
                    continue
                new_branches = sorted(current_branches + [v])
                if try_accept(new_branches):
                    current_branches = list(best_result[0])
                    improved = True
                    break

    # 4. Broader randomized search over high-degree vertices
    if time_ok():
        high_deg = sorted(range(n), key=lambda x: -degrees[x])[:20]
        for size in range(min(6, len(high_deg)), 1, -1):
            if not time_ok():
                break
            for _ in range(20):
                if not time_ok():
                    break
                subset = random.sample(high_deg, size)
                if try_accept(subset):
                    break

    # 5. Fallback
    if best_result is None or best_k < 2:
        u, v = edges[0]
        best_result = ([int(u), int(v)], [[int(u), int(v)]])

    branches = [int(b) for b in best_result[0]]
    paths_out = [[int(v) for v in p] for p in best_result[1]]

    return {
        'branches': branches,
        'paths': paths_out
    }
