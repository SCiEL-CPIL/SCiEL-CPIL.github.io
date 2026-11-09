import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin vs 90s kill

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

    # ---------- Dinic max-flow with vertex splitting ----------
    # Node indexing: vin=2v, vout=2v+1, SRC=2n, SNK=2n+1
    SRC = 2 * n
    SNK = 2 * n + 1
    TOTAL = 2 * n + 2
    INF = 10**9

    def dinic(branches):
        """Verify a branch set with max-flow vertex splitting.
        Returns list of (path, pair) or None. A branch set is feasible iff
        max flow == k(k-1)/2, where each unit of flow is a branch->branch path
        with interior vertex capacity 1 (enforces pairwise interior-disjointness)."""
        k = len(branches)
        if k < 2:
            return None
        need = k * (k - 1) // 2
        branches_set = set(branches)

        cap = [[0] * TOTAL for _ in range(TOTAL)]

        def add_edge(u, v, c):
            cap[u][v] += c
            cap[v][u] += 0  # keep reverse entry present

        # vertex split
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            c = INF if v in branches_set else 1
            cap[vin][vout] += c
        # graph edges
        for u in range(n):
            for v in adj_list[u]:
                if u < v:
                    cap[2 * u + 1][2 * v] += INF
                    cap[2 * v + 1][2 * u] += INF
        # source/sink
        for b in branches:
            cap[SRC][2 * b] += INF
            cap[2 * b + 1][SNK] += INF

        flow = 0
        level = [-1] * TOTAL
        it = [0] * TOTAL

        def bfs():
            for i in range(TOTAL):
                level[i] = -1
            dq = deque([SRC])
            level[SRC] = 0
            while dq:
                u = dq.popleft()
                lu = level[u]
                row = cap[u]
                for v in range(TOTAL):
                    if level[v] < 0 and row[v] > 0:
                        level[v] = lu + 1
                        if v == SNK:
                            return True
                        dq.append(v)
            return level[SNK] >= 0

        def dfs(u, f):
            if u == SNK:
                return f
            row = cap[u]
            rev = cap
            while it[u] < TOTAL:
                v = it[u]
                if row[v] > 0 and level[v] == level[u] + 1:
                    ret = dfs(v, f if f < row[v] else row[v])
                    if ret > 0:
                        row[v] -= ret
                        rev[v][u] += ret
                        return ret
                it[u] += 1
            return 0

        while bfs():
            it = [0] * TOTAL
            while True:
                if not time_ok():
                    return None
                f = dfs(SRC, INF)
                if f == 0:
                    break
                flow += f
                if flow > need:
                    break
            if flow > need:
                break

        if flow < need:
            return None

        # extract paths from source out-edges
        src_row = cap[SRC]
        paths = []
        for b in branches:
            cnt = need - b  # units from b to later branches
            if cnt <= 0:
                continue
            cur = 2 * b
            for _ in range(cnt):
                row = cap[cur]
                nxt = -1
                for v in range(TOTAL):
                    if row[v] > 0:
                        nxt = v
                        break
                if nxt < 0:
                    return None
                row[nxt] -= 1
                cap[nxt][cur] += 1
                cur = nxt
                if cur == SNK:
                    break
            # cur == SNK
            # trace back
            path = [SNK]
            c = SNK
            while c != SRC:
                cprev = -1
                for v in range(TOTAL):
                    if cap[c][v] > 0:
                        cprev = v
                        break
                if cprev < 0:
                    return None
                cap[c][cprev] -= 1
                cap[cprev][c] += 1
                path.append(cprev)
                c = cprev
            path.reverse()
            path = path[1:-1]  # drop SRC and SNK
            verts = []
            for x in path:
                if x % 2 == 0:
                    if not verts or verts[-1] != x // 2:
                        verts.append(x // 2)
            if len(verts) < 2:
                return None
            a, b2 = verts[0], verts[-1]
            if a not in branches_set or b2 not in branches_set or a == b2:
                return None
            # ensure a<b to form unordered pair
            if a > b2:
                verts = verts[::-1]
                a, b2 = b2, a
            paths.append((verts, (a, b2)))
        if len(paths) != need:
            return None
        # sanity: every unordered pair present exactly once
        seen = set()
        for _, pr in paths:
            if pr in seen:
                return None
            seen.add(pr)
        if len(seen) != need:
            return None
        return paths

    # ---------- Max clique (Bron-Kerbosch with pivot) ----------
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
            pc = list(Pset | X) if Pset else list(X)
            if not pc:
                return
            u = max(pc, key=lambda v: len(adj[v] & Pset))
            cands = [v for v in Pset if v not in adj[u]]
            cands.sort(key=lambda x: -degrees[x])
            for v in cands:
                if not time_ok():
                    return
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset = Pset - {v}
                X = X | {v}

        bk(frozenset(), frozenset(range(n)), frozenset())
        return best

    # ---------- Greedy grow using flow verification ----------
    def greedy_grow(seed_branches):
        branches = sorted(seed_branches)
        improved = True
        while improved and time_ok():
            improved = False
            for v in range(n):
                if v in set(branches):
                    continue
                cand = sorted(branches + [v])
                if dinic(cand) is not None:
                    branches = cand
                    improved = True
                    break
        return branches

    # ---------- Main ----------
    best_branches = None
    best_paths = None

    # 1. Try max clique
    mc = find_max_clique()
    mc_sorted = sorted(mc)
    if len(mc_sorted) >= 2:
        paths = dinic(mc_sorted)
        if paths is not None:
            best_branches, best_paths = mc_sorted, paths
        else:
            # grow from a feasible clique subset
            for size in range(min(len(mc_sorted), 4), 1, -1):
                if not time_ok():
                    break
                for combo in combinations(mc_sorted, size):
                    if not time_ok():
                        break
                    sub = sorted(combo)
                    paths = dinic(sub)
                    if paths is not None:
                        grown = greedy_grow(sub)
                        gp = dinic(grown)
                        if gp is not None:
                            best_branches, best_paths = grown, gp
                        else:
                            best_branches, best_paths = sub, paths
                        break
                if best_branches is not None:
                    break

    # 2. If still nothing, grow from high-degree seeds
    if best_branches is None or len(best_branches) < 2:
        high = sorted(range(n), key=lambda x: -degrees[x])
        for i in range(len(high)):
            for j in range(i + 1, len(high)):
                if not time_ok():
                    break
                pair = sorted([high[i], high[j]])
                paths = dinic(pair)
                if paths is not None:
                    grown = greedy_grow(pair)
                    gp = dinic(grown)
                    if gp is not None:
                        best_branches, best_paths = grown, gp
                    else:
                        best_branches, best_paths = pair, paths
                    break
            if best_branches is not None:
                break

    # 3. Fallback: first edge
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = sorted([u, v])
        best_paths = [([int(u), int(v)], (min(u, v), max(u, v)))]

    # Build output: one path per unordered pair, ordered by (min,max)
    k = len(best_branches)
    pairs = [(best_branches[i], best_branches[j]) for i in range(k) for j in range(i + 1, k)]
    path_by_pair = {}
    for p, pr in best_paths:
        path_by_pair[pr] = p
    paths_out = []
    for a, b in pairs:
        key = (a, b) if a < b else (b, a)
        p = path_by_pair[key]
        if p[0] == a:
            paths_out.append(p)
        else:
            paths_out.append(p[::-1])

    branches = [int(b) for b in best_branches]
    paths_out = [[int(v) for v in p] for p in paths_out]

    return {
        'branches': branches,
        'paths': paths_out
    }
