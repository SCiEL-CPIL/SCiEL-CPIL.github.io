"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    deadline = time.time() + 80  # safety margin before the 90s kill
    n = P['n']

    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    deg = [len(adj[i]) for i in range(n)]

    def time_left():
        return time.time() < deadline

    # ------------------------------------------------------------------
    # Shortest simple path from src to dst avoiding `blocked` interior
    # vertices (src/dst are always allowed).
    # ------------------------------------------------------------------
    def bfs_path(src, dst, blocked):
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for v in adj_list[u]:
                if v in parent:
                    continue
                if v != dst and v in blocked:
                    continue
                parent[v] = u
                queue.append(v)
        return None

    # ------------------------------------------------------------------
    # Bron-Kerbosch with greedy-coloring bound -> maximum clique.
    # ------------------------------------------------------------------
    def max_clique():
        best = [0]
        best_set = None

        def color_sort(R, P):
            order = []
            colors = {}
            available = list(P)
            col = 0
            while available:
                col += 1
                chosen = []
                for v in available:
                    if all(u in colors for u in available if u in adj[v]):
                        chosen.append(v)
                if not chosen:
                    chosen = [available[0]]
                for v in chosen:
                    colors[v] = col
                    order.append(v)
                available = [v for v in available if v not in colors]
            return order, colors

        def bk(R, P, X):
            if not time_left():
                return
            if not P and not X:
                if len(R) > best[0]:
                    best[0] = len(R)
                    best_set = list(R)
                return
            order, colors = color_sort(R, P)
            for i in range(len(order) - 1, -1, -1):
                v = order[i]
                if len(R) + colors[v] <= best[0]:
                    return
                bk(R | {v}, P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return best_set

    # ------------------------------------------------------------------
    # Routing: given a branch set, find pairwise-disjoint interior paths
    # for every unordered pair.  Direct edges cost nothing; non-direct
    # pairs get a shortest path whose interior is disjoint from all others
    # and from every branch.
    # ------------------------------------------------------------------
    def route(branches):
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))

        # feasibility: every non-adjacent pair must have a common
        # non-branch neighbour to route through (necessary condition)
        non_direct = []
        for u, v in pairs:
            if v not in adj[u]:
                ok = False
                for w in adj[u]:
                    if w not in branch_set and w in adj[v]:
                        ok = True
                        break
                if not ok:
                    return None
                non_direct.append((u, v))

        used = set()
        assign = {}

        # order: fewest common neighbours first (most constrained)
        non_direct.sort(key=lambda p: sum(1 for w in adj[p[0]] if w not in branch_set and w in adj[p[1]]))

        # bounded backtracking to repair a greedy failure
        def backtrack(idx):
            if not time_left():
                return False
            if idx == len(non_direct):
                return True
            u, v = non_direct[idx]
            blocked = branch_set | used
            blocked.discard(u)
            blocked.discard(v)
            path = bfs_path(u, v, blocked)
            if path is None:
                return False
            interior = path[1:-1]
            if len(interior) != len(set(interior)):
                return False
            for w in interior:
                if w in branch_set:
                    return False
            for w in interior:
                used.add(w)
            assign[(u, v)] = interior
            if backtrack(idx + 1):
                return True
            for w in interior:
                used.discard(w)
            del assign[(u, v)]
            return False

        if not backtrack(0):
            return None

        result = {}
        for u, v in pairs:
            if v in adj[u]:
                result[(u, v)] = [u, v]
            else:
                result[(u, v)] = [u] + assign[(u, v)] + [v]
        return result

    def try_expand(start):
        branches = list(start)
        paths = route(branches)
        if paths is None:
            return None
        cur = {k: list(p) for k, p in paths.items()}
        cur_len = len(branches)
        candidates = [v for v in range(n) if v not in branches]
        candidates.sort(key=lambda v: -deg[v])
        for cand in candidates:
            if not time_left():
                break
            new_branches = branches + [cand]
            new_paths = route(new_branches)
            if new_paths is not None:
                branches = new_branches
                cur = {k: list(p) for k, p in new_paths.items()}
                cur_len = len(branches)
        if cur_len >= 2:
            return branches, cur
        return None

    # ------------------------------------------------------------------
    # Drive: start from the maximum clique (densest backbone), then
    # greedily expand; also probe a few other high-degree seeds in case
    # the max clique cannot be expanded as far.
    # ------------------------------------------------------------------
    best = None
    best_len = 0

    mc = max_clique()
    seeds = []
    if mc:
        seeds.append(mc)
    # a couple of alternative high-degree seeds
    for seed in [list(range(n))]:
        seed.sort(key=lambda v: -deg[v])
        for k in (5, 4, 3):
            seeds.append(seed[:k])
    # dedupe
    seen = set()
    uniq = []
    for s in seeds:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    for s in uniq:
        if not time_left():
            break
        res = try_expand(s)
        if res is not None:
            branches, paths = res
            if len(branches) > best_len:
                best_len = len(branches)
                best = (branches, paths)

    if best is None or best_len < 2:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    branches, paths = best
    result_branches = [int(b) for b in branches]
    result_paths = []
    for (u, v), path in paths.items():
        result_paths.append([int(x) for x in path])

    return {'branches': result_branches, 'paths': result_paths}
# EVOLVE-BLOCK-END
