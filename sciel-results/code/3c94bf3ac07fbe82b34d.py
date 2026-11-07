"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0
    n = int(P['n'])
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def find_path(src, dst, forbidden):
        """Find shortest simple path from src to dst where internal vertices avoid forbidden."""
        if src == dst:
            return None
        prev = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb in prev:
                    continue
                if nb != dst and nb in forbidden:
                    continue
                prev[nb] = cur
                if nb == dst:
                    path = []
                    x = dst
                    while x is not None:
                        path.append(x)
                        x = prev[x]
                    path.reverse()
                    return path
                q.append(nb)
        return None

    def try_add(branches, used, v, adj):
        """Try to add vertex v to branches, returning new used dict or None."""
        new_used = dict(used)
        forbidden = set(branches)
        for u in branches:
            p = find_path(u, v, forbidden)
            if p is None:
                return None
            new_used[(min(u, v), max(u, v))] = p
            forbidden.update(p)  # add internal vertices to forbidden
        return new_used

    # Greedy seed
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    branches = []
    used = {}
    for v in order:
        new_used = try_add(branches, used, v, adj)
        if new_used is not None:
            branches.append(v)
            used = new_used

    best_k = len(branches)
    best = list(branches)
    best_paths = dict(used)

    # Backtracking DFS to find larger clique subdivisions
    def dfs(branches, used, idx, budget):
        nonlocal best_k, best, best_paths
        k = len(branches)
        if k > best_k:
            best_k = k
            best = list(branches)
            best_paths = dict(used)
        if k >= n - 1 or time.time() > budget:
            return
        if k + (n - 1 - idx) <= best_k:
            return
        for v in range(idx, n):
            if v in branches:
                continue
            new_used = try_add(branches, used, v, adj)
            if new_used is not None:
                branches.append(v)
                dfs(branches, new_used, v + 1, budget)
                branches.pop()

    dfs(list(branches), dict(used), 0, DEADLINE - 10.0)

    # Build final result
    if best is None or len(best) < 2:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    final_branches = list(best)
    path_list = []
    for u, v in combinations(final_branches, 2):
        path_list.append([int(x) for x in best_paths[(min(u, v), max(u, v))]])

    return {'branches': [int(x) for x in final_branches], 'paths': path_list}
# EVOLVE-BLOCK-END
