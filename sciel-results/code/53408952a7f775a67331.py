"""Maximum pinned-graph clique subdivision via max-clique + matching feasibility."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t0 = time.time()
    DEADLINE = t0 + 55.0
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)

    def bfs_paths(s, t, banned, limit):
        """All simple s-t paths using only vertices in `banned` as interiors."""
        if s == t:
            return []
        if not banned or t not in adj[s]:
            return []
        res = []
        q = deque([(s, frozenset())])
        seen = {s}
        while q and len(res) < limit:
            v, pathset = q.popleft()
            if v == t:
                res.append(pathset)
                continue
            for w in adj[v]:
                if w == s:
                    continue
                if w in seen:
                    continue
                if w == t:
                    q.append((w, pathset))
                    seen.add(w)
                else:
                    if w in banned:
                        q.append((w, pathset | frozenset((w,))))
                        seen.add(w)
        return res

    def feasible(branches):
        """Return dict {pair:(i,j): [v0..vk]} of internally-disjoint paths, or None."""
        from itertools import combinations
        B = list(branches)
        pairs = list(combinations(range(len(B)), 2))
        if not pairs:
            return {}
        banned = set(range(n))
        for b in B:
            banned.discard(b)
        paths = {}
        for (i, j) in pairs:
            ps = bfs_paths(B[i], B[j], banned, 200)
            if not ps:
                return None
            paths[(i, j)] = ps
        # Bipartite matching: left = pairs, right = interior vertices, edges = path uses it
        left = pairs
        right = list(banned)
        ridx = {v: k for k, v in enumerate(right)}
        inc = [[] for _ in range(len(left))]
        for (i, j), ps in paths.items():
            li = left.index((i, j))
            for ps_ in ps:
                for v in ps_:
                    inc[li].append(ridx[v])
        # Hopcroft-Karp
        INF = 10 ** 9
        dist = [0] * len(left)
        matchL = [-1] * len(left)
        matchR = [-1] * len(right)

        def bfs():
            dq = deque()
            for u in range(len(left)):
                if matchL[u] == -1:
                    dist[u] = 0
                    dq.append(u)
                else:
                    dist[u] = INF
            found = False
            while dq:
                u = dq.popleft()
                for r in inc[u]:
                    v = matchR[r]
                    if v == -1:
                        found = True
                    elif dist[v] == INF:
                        dist[v] = dist[u] + 1
                        dq.append(v)
            return found

        def dfs(u):
            for r in inc[u]:
                v = matchR[r]
                if v == -1 or (dist[v] == dist[u] + 1 and dfs(v)):
                    matchL[u] = r
                    matchR[r] = u
                    return True
            dist[u] = INF
            return False

        matching = 0
        while bfs():
            for u in range(len(left)):
                if matchL[u] == -1 and dfs(u):
                    matching += 1
        if matching != len(left):
            return None
        # reconstruct
        out = {}
        for (i, j), ps in paths.items():
            li = left.index((i, j))
            r = matchL[li]
            v = right[r]
            chosen = None
            for ps_ in ps:
                if v in ps_:
                    chosen = ps_
                    break
            if chosen is None:
                return None
            seq = [B[i]] + list(chosen) + [B[j]]
            out[(i, j)] = seq
        return out

    def max_clique():
        best = 0
        best_set = set()
        stack = [(list(range(n)), 0)]
        while stack:
            cand, sz = stack.pop()
            if sz + len(cand) <= best:
                continue
            if not cand:
                if sz > best:
                    best = sz
                    best_set = set()
                continue
            v = cand[0]
            rest = cand[1:]
            # branch: include v
            newcand = [w for w in rest if w in adj[v]]
            stack.append((newcand, sz + 1))
            # branch: exclude v
            stack.append((rest, sz))
        return best

    # Trivial fallback: first edge
    fallback = {'branches': [int(P['edges'][0][0]), int(P['edges'][0][1])],
                'paths': [[int(P['edges'][0][0]), int(P['edges'][0][1])]]}

    # 1) exact max clique (trivial subdivision)
    try:
        mc = max_clique()
    except Exception:
        mc = 2

    # 2) search for the largest feasible branch set, descending size
    found = None
    order = list(range(n))
    from itertools import combinations
    for k in range(mc, 1, -1):
        if time.time() > DEADLINE:
            break
        for combo in combinations(order, k):
            if time.time() > DEADLINE:
                break
            res = feasible(combo)
            if res is not None:
                found = (list(combo), res)
                break
        if found is not None:
            break

    if found is not None:
        branches, res = found
        branches = [int(b) for b in branches]
        paths = []
        for (i, j), seq in res.items():
            paths.append([int(x) for x in seq])
        return {'branches': branches, 'paths': paths}

    # 3) fallback: a clique of size mc as trivial subdivision
    # re-find a concrete clique of size mc via greedy
    clique = []
    cand = list(range(n))
    while cand and len(clique) < mc:
        if time.time() > DEADLINE:
            break
        v = cand[0]
        clique.append(v)
        cand = [w for w in cand[1:] if w in adj[v]]
    if len(clique) < 2:
        return fallback
    branches = [int(b) for b in clique]
    paths = []
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            paths.append([int(clique[i]), int(clique[j])])
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
