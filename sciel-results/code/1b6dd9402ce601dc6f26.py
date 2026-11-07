import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # All-pairs shortest path distances
    INF = n + 1
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        for j in adj[i]:
            dist[i][j] = 1
    for k in range(n):
        for i in range(n):
            if dist[i][k] >= INF:
                continue
            for j in range(n):
                if dist[k][j] >= INF:
                    continue
                nd = dist[i][k] + dist[k][j]
                if nd < dist[i][j]:
                    dist[i][j] = nd

    # ---- Max-flow feasibility check for vertex-disjoint interior paths ----
    def flow_feasible(branches):
        bset = set(branches)
        k = len(branches)
        if k < 2:
            return True
        required = k * (k - 1) // 2
        S = 2 * n
        T = 2 * n + 1
        total = 2 * n + 2
        adj_f = [[] for _ in range(total)]

        def add_edge(u, v, cap):
            adj_f[u].append([v, cap, len(adj_f[v])])
            adj_f[v].append([u, 0, len(adj_f[u]) - 1])

        for v in range(n):
            if v in bset:
                add_edge(2 * v, 2 * v + 1, 10**9)
            else:
                add_edge(2 * v, 2 * v + 1, 1)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    add_edge(2 * u + 1, 2 * v, 10**9)
                    add_edge(2 * v + 1, 2 * u, 10**9)
        for b in branches:
            add_edge(S, 2 * b, 10**9)
            add_edge(2 * b + 1, T, 10**9)

        # Dinic's max-flow
        from collections import deque as _dq
        level = [0] * total
        it = [0] * total

        def bfs_level():
            for i in range(total):
                level[i] = -1
            level[S] = 0
            q = _dq([S])
            while q:
                u = q.popleft()
                for e in adj_f[u]:
                    if e[1] > 0 and level[e[0]] < 0:
                        level[e[0]] = level[u] + 1
                        q.append(e[0])
            return level[T] >= 0

        def dfs(u, f):
            if u == T:
                return f
            while it[u] < len(adj_f[u]):
                e = adj_f[u][it[u]]
                v = e[0]
                if e[1] > 0 and level[u] < level[v]:
                    d = dfs(v, min(f, e[1]))
                    if d > 0:
                        e[1] -= d
                        adj_f[v][e[2]][1] += d
                        return d
                it[u] += 1
            return 0

        total_flow = 0
        while bfs_level():
            for i in range(total):
                it[i] = 0
            while True:
                f = dfs(S, 10**18)
                if f == 0:
                    break
                total_flow += f
        return total_flow >= required

    # ---- Find paths via backtracking with good ordering ----
    def find_paths(branches):
        if len(branches) < 2:
            return None
        bset = set(branches)
        pairs = sorted(combinations(sorted(branches), 2))
        # Order: direct edges first, then by shortest-path length
        pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else dist[p[0]][p[1]]))

        paths = {}
        used = set()

        def bsearch(idx):
            if idx == len(pairs):
                return True
            if time.time() > deadline:
                return False
            u, v = pairs[idx]
            if v in adj[u]:
                paths[(u, v)] = [u, v]
                ok = bsearch(idx + 1)
                if ok:
                    return True
                del paths[(u, v)]
                return False

            forbidden = bset - {u, v}
            forbidden |= used

            # BFS to find shortest path avoiding forbidden
            parent = {u: None}
            q = deque([u])
            found = False
            while q:
                if time.time() > deadline:
                    return False
                cur = q.popleft()
                if cur == v:
                    path = []
                    node = v
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    paths[(u, v)] = path
                    new_used = set(path[1:-1])
                    if not (new_used & forbidden):
                        used.update(new_used)
                        if bsearch(idx + 1):
                            return True
                        used -= new_used
                    del paths[(u, v)]
                    return False
                for nb in sorted(adj[cur], key=lambda x: dist[x][v]):
                    if nb not in parent and nb not in forbidden:
                        parent[nb] = cur
                        q.append(nb)
            return False

        if bsearch(0):
            return paths
        return None

    # ---- Bron-Kerbosch with pivot to find max clique ----
    best_clique = []

    def bronk(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return

    bronk(set(), set(range(n)), set())

    # ---- Greedy expansion: try adding vertices one at a time ----
    current_branches = list(best_clique)
    current_paths = find_paths(current_branches)
    if current_paths is None:
        # Fallback: use just the clique with direct edges
        current_paths = {}
        for u, v in combinations(sorted(current_branches), 2):
            current_paths[(u, v)] = [u, v]

    best_branches = list(current_branches)
    best_paths = dict(current_paths)

    remaining = [v for v in range(n) if v not in set(current_branches)]
    remaining.sort(key=lambda v: -len(adj[v]))

    improved = True
    while improved:
        improved = False
        for v in remaining[:]:
            if time.time() > deadline - 5:
                break
            test = current_branches + [v]
            if flow_feasible(test):
                test_paths = find_paths(test)
                if test_paths is not None:
                    current_branches = test
                    current_paths = test_paths
                    remaining.remove(v)
                    improved = True
                    best_branches = list(current_branches)
                    best_paths = dict(current_paths)

    # ---- Local search: try swaps to potentially find larger set ----
    if time.time() < deadline - 15:
        for _ in range(2):
            swap_improved = False
            cur_set = set(current_branches)
            for i in range(len(current_branches)):
                if time.time() > deadline - 5:
                    break
                for v in remaining:
                    if time.time() > deadline - 5:
                        break
                    test = [b for j, b in enumerate(current_branches) if j != i] + [v]
                    if flow_feasible(test):
                        test_paths = find_paths(test)
                        if test_paths is not None:
                            current_branches = test
                            current_paths = test_paths
                            remaining.remove(v)
                            remaining.append(current_branches[i] if i < len(current_branches) else test[0])
                            swap_improved = True
                            best_branches = list(current_branches)
                            best_paths = dict(current_paths)
                            break
                if swap_improved:
                    break
            if not swap_improved:
                break

    # Use best found
    if len(best_branches) >= 2:
        branches = sorted(best_branches)
        # Verify and build result
        paths_list = []
        valid = True
        for u, v in combinations(branches, 2):
            p = best_paths.get((u, v))
            if p is None:
                p = [u, v]
            paths_list.append([int(x) for x in p])

        result_branches = [int(b) for b in branches]
        return {'branches': result_branches, 'paths': paths_list}

    # Absolute fallback: find any valid 2-branch solution
    # Pick two adjacent vertices
    for u, v in P['edges']:
        return {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }

    # Should not reach here
    return {'branches': [0, 1], 'paths': [[0, 1]]}
