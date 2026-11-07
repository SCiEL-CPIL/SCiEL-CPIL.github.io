"""Combined program: flow-based vertex-disjoint path oracle + greedy set extension."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque
import networkx as nx
import networkx.algorithms.flow as nxflow

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin under 90s limit

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def check_time():
        return time.time() < deadline

    # ---- Flow-based feasibility oracle (vertex-splitting, capacity 1) ----
    def flow_oracle(branches):
        """Return (feasible, paths_dict) for a candidate branch set.
        Exact: feasible iff a valid subdivision exists (vertex-disjoint paths
        for every unordered pair, interiors disjoint, avoiding all branches)."""
        k = len(branches)
        if k < 2:
            return False, None
        bset = set(branches)
        # Split vertices: in=2v, out=2v+1
        G = nx.DiGraph()
        for v in range(n):
            G.add_edge(2 * v, 2 * v + 1, capacity=1, length=1)
        for u, v in edges:
            G.add_edge(2 * u + 1, 2 * v, capacity=1, length=1)
            G.add_edge(2 * v + 1, 2 * u, capacity=1, length=1)
        S = 2 * n
        T = 2 * n + 1
        G.add_edge(S, T, capacity=1, length=1)
        for b in branches:
            G.add_edge(S, 2 * b, capacity=1, length=1)
            G.add_edge(2 * b + 1, T, capacity=1, length=1)
        try:
            flow_value, flow_dict = nxflow.max_flow_min_cost(G, S, T)
        except Exception:
            return False, None
        if flow_value < k * (k - 1) // 2:
            return False, None
        # Reconstruct paths from flow
        paths = {}
        used = set()
        for b in branches:
            for nb in G[2 * b + 1]:
                if nb == T:
                    continue
                if flow_dict[2 * b + 1][nb] < 1:
                    continue
                # walk out of b
                path = [b]
                cur = nb
                while True:
                    curv = cur // 2
                    path.append(curv)
                    nxt = None
                    for nb2 in G[cur]:
                        if nb2 == T:
                            continue
                        if flow_dict[cur][nb2] >= 1:
                            nxt = nb2
                            break
                    if nxt is None:
                        break
                    cur = nxt
                if len(path) >= 2:
                    a, c = path[0], path[-1]
                    if a == c:
                        continue
                    key = (a, c) if a < c else (c, a)
                    if key not in used:
                        used.add(key)
                        paths[key] = path
        if len(paths) != k * (k - 1) // 2:
            return False, None
        return True, paths

    def is_valid(branches, paths_dict):
        if len(branches) < 2:
            return False
        if len(set(branches)) != len(branches):
            return False
        bset = set(branches)
        for u, v in combinations(branches, 2):
            if (u, v) not in paths_dict and (v, u) not in paths_dict:
                return False
        used_interiors = set()
        for key, path in paths_dict.items():
            if len(path) < 2:
                return False
            u, v = key
            if path[0] not in (u, v) or path[-1] not in (u, v):
                return False
            if path[0] == path[-1]:
                return False
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return False
            for node in path[1:-1]:
                if node in bset:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(path[1:-1])
        return True

    # ---- Flow-based repair: free interior vertices, re-route ----
    def bfs_path(u, v, forbidden):
        if u in forbidden or v in forbidden:
            return None
        if u == v:
            return [u, v]
        visited = {u: None}
        q = deque([u])
        while q:
            node = q.popleft()
            if node == v:
                path = [v]
                while path[-1] != u:
                    path.append(visited[path[-1]])
                path.reverse()
                return path
            for nb in adj[node]:
                if nb not in visited and nb not in forbidden:
                    visited[nb] = node
                    q.append(nb)
        return None

    def repair(branches, paths):
        """Try to reduce interior footprint by re-routing with freed vertices."""
        bset = set(branches)
        for _ in range(8):
            if not check_time():
                break
            used = set()
            for p in paths.values():
                used.update(p[1:-1])
            if not used:
                break
            freed = set(used)
            found = False
            for key in list(paths.keys()):
                u, v = key
                forbid = set(bset) - {u, v}
                p = bfs_path(u, v, forbid)
                if p is not None:
                    paths[key] = p
                    found = True
            if not found:
                break
            new_used = set()
            for p in paths.values():
                new_used.update(p[1:-1])
            if new_used == used:
                break
            used = new_used
        return paths

    # ---- Greedy set extension using the flow oracle ----
    def greedy_extend(start, max_iters=60):
        cur = list(start)
        ok, paths = flow_oracle(cur)
        if not ok:
            return [], {}
        for _ in range(max_iters):
            if not check_time():
                break
            cands = [i for i in range(n) if i not in cur]
            cands.sort(key=lambda x: len(adj[x]), reverse=True)
            added = False
            for nb in cands[:14]:
                new = cur + [nb]
                ok2, p2 = flow_oracle(new)
                if ok2 and is_valid(new, p2):
                    cur = new
                    paths = p2
                    added = True
                    break
            if not added:
                break
        return cur, paths

    # ---- Max clique (Bron-Kerbosch with pivot) ----
    def find_max_clique():
        best = []
        def bk(R, Pset, X):
            nonlocal best
            if not check_time():
                return
            if not Pset and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if Pset:
                pivot = max(Pset, key=lambda v: len(adj[v] & (Pset | X)))
            else:
                pivot = None
            cands = list(Pset)
            if pivot is not None:
                cands = [v for v in cands if v not in adj[pivot]]
            for v in cands:
                if not check_time():
                    return
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset.discard(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return best

    best_branches = []
    best_paths = {}

    def consider(branches, paths):
        nonlocal best_branches, best_paths
        if len(branches) > len(best_branches):
            best_branches = list(branches)
            best_paths = dict(paths)

    # Strategy 1: max clique seed, shrink to feasible size, then extend
    clique = find_max_clique()
    if len(clique) >= 2:
        clique_sorted = sorted(clique, key=lambda x: len(adj[x]), reverse=True)
        for size in range(min(len(clique_sorted), 20), 1, -1):
            if not check_time():
                break
            subset = clique_sorted[:size]
            ok, paths = flow_oracle(subset)
            if ok and is_valid(subset, paths):
                consider(subset, paths)
                cur, ext = greedy_extend(subset, max_iters=40)
                consider(cur, ext)
                break
        for trial in range(30):
            if not check_time():
                break
            if len(clique_sorted) <= len(best_branches):
                break
            subset = random.sample(clique_sorted,
                                   min(len(clique_sorted), len(best_branches) + 1))
            ok, paths = flow_oracle(subset)
            if ok:
                cur, ext = greedy_extend(subset, max_iters=35)
                consider(cur, ext)

    # Strategy 2: greedy from high-degree edges
    if len(best_branches) < 5:
        sorted_edges = sorted(edges, key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
        for u, v in sorted_edges[:16]:
            if not check_time():
                break
            cur, paths = greedy_extend([u, v], max_iters=45)
            consider(cur, paths)

    # Strategy 3: local search (swap / add)
    if len(best_branches) >= 3:
        improved = True
        while improved and check_time():
            improved = False
            for trial in range(25):
                if not check_time():
                    break
                rem_idx = random.randint(0, len(best_branches) - 1)
                remaining = [b for i, b in enumerate(best_branches) if i != rem_idx]
                cands = [i for i in range(n) if i not in remaining]
                random.shuffle(cands)
                for nb in cands[:8]:
                    new = remaining + [nb]
                    ok, paths = flow_oracle(new)
                    if ok and is_valid(new, paths) and len(new) > len(best_branches):
                        consider(new, paths)
                        improved = True
                        break

    # Strategy 4: random restarts
    for restart in range(12):
        if not check_time():
            break
        if not edges:
            break
        u, v = random.choice(edges)
        cur, paths = greedy_extend([u, v], max_iters=40)
        consider(cur, paths)

    # Final: repair best solution to reduce interior footprint (optional)
    if len(best_branches) >= 2:
        rep = repair(list(best_branches), dict(best_paths))
        if is_valid(best_branches, rep):
            best_paths = rep

    # Build output
    if len(best_branches) < 2:
        if edges:
            u, v = edges[0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    final_paths = []
    for u, v in combinations(sorted(best_branches), 2):
        key = (u, v)
        if key in best_paths:
            final_paths.append([int(x) for x in best_paths[key]])
        elif (v, u) in best_paths:
            final_paths.append([int(x) for x in best_paths[(v, u)]])

    return {
        'branches': [int(x) for x in best_branches],
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
