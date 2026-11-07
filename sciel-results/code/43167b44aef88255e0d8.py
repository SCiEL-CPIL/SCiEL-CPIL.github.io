"""Rewritten: clique-subdivision via min-cut feasibility + Bron-Kerbosch + greedy growing."""

import time
import collections

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = int(P['n'])
ADJ = [set() for _ in range(N)]
for _a, _b in P['edges']:
    ADJ[_a].add(_b)
    ADJ[_b].add(_a)

def _build_network(branches):
    branches_set = set(branches)
    k = len(branches)
    S = 0
    T = 1
    bn_index = {}
    for pos, v in enumerate(branches):
        bn_index[v] = 2 + pos
    IN = lambda v: 2 + k + v
    V = 2 + k + N
    cap = [0] * (V * V)
    for v in branches:
        cap[S * V + bn_index[v]] = 1
        cap[bn_index[v] * V + T] = 1
    for i in range(k):
        for j in range(i + 1, k):
            a, b = branches[i], branches[j]
            if b in ADJ[a]:
                cap[bn_index[a] * V + bn_index[b]] = 1
                cap[bn_index[b] * V + bn_index[a]] = 1
    for u in branches:
        for v in ADJ[u]:
            if v in branches_set:
                continue
            cap[bn_index[u] * V + IN(v)] = 1
            cap[IN(v) * V + bn_index[u]] = 1
    for u in range(N):
        if u in branches_set:
            continue
        for v in ADJ[u]:
            if v in branches_set:
                continue
            if u >= v:
                continue
            cap[IN(u) * V + IN(v)] = 1
            cap[IN(v) * V + IN(u)] = 1
    return S, T, V, cap, bn_index

def _feasible(branches, deadline):
    k = len(branches)
    need = k * (k - 1) // 2
    if need == 0:
        return False, None
    S, T, V, cap, bn_index = _build_network(branches)
    branches_set = set(branches)
    flow = 0
    flow_paths = []
    while flow < need:
        prev = [-1] * V
        prev[S] = S
        q = collections.deque([S])
        found = False
        while q:
            u = q.popleft()
            if u == T:
                found = True
                break
            row = u * V
            for v in range(V):
                if prev[v] == -1 and cap[row + v] > 0:
                    prev[v] = u
                    q.append(v)
        if not found:
            break
        v = T
        path = []
        while v != S:
            u = prev[v]
            path.append((u, v))
            v = u
        path.reverse()
        for (u, v) in path:
            cap[u * V + v] -= 1
            cap[v * V + u] += 1
        flow_paths.append(path)
        flow += 1
    if flow < need:
        return False, None
    # Extract vertex paths
    def node_to_vertex(node):
        if 2 <= node < 2 + k:
            return branches[node - 2]
        elif 2 + k <= node < 2 + k + N:
            return node - (2 + k)
        return None
    paths_dict = {}
    for fp in flow_paths:
        assert fp[0][0] == S and fp[-1][1] == T
        nodes = [fp[0][1]] + [fp[i][1] for i in range(1, len(fp) - 1)]
        a = node_to_vertex(nodes[0])
        b = node_to_vertex(nodes[-1])
        if a is None or b is None:
            return False, None
        mid = nodes[1:-1]
        vertex_path = [a] + [node_to_vertex(x) for x in mid] + [b]
        for x in vertex_path:
            if x in branches_set and x not in (a, b):
                return False, None
        for i in range(len(vertex_path) - 1):
            if vertex_path[i + 1] not in ADJ[vertex_path[i]]:
                return False, None
        key = (min(a, b), max(a, b))
        if key in paths_dict:
            return False, None
        paths_dict[key] = vertex_path
    if len(paths_dict) != need:
        return False, None
    return True, paths_dict

def _verify(branches, paths_dict):
    k = len(branches)
    need = k * (k - 1) // 2
    if len(paths_dict) != need:
        return False
    branches_set = set(branches)
    interiors = set()
    for (a, b), p in paths_dict.items():
        if len(p) < 2:
            return False
        if set([p[0], p[-1]]) != set([a, b]):
            return False
        for i in range(len(p) - 1):
            if p[i + 1] not in ADJ[p[i]]:
                return False
        for x in p[1:-1]:
            if x in branches_set:
                return False
            if x in interiors:
                return False
            interiors.add(x)
    return True

def _bron_kerbosch(adj, R, P_set, X, deadline, on_clique):
    if not P_set and not X:
        on_clique(list(R))
        return
    union = P_set | X
    if union:
        pivot = max(union, key=lambda v: len(adj[v] & union))
        candidates = [v for v in (P_set - adj.get(pivot, set()))]
    else:
        candidates = list(P_set)
    for v in candidates:
        if time.time() > deadline:
            return
        R.add(v)
        new_P = P_set & adj[v]
        new_X = X & adj[v]
        _bron_kerbosch(adj, R, new_P, new_X, deadline, on_clique)
        R.discard(v)
        P_set.discard(v)
        X.add(v)

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0
    n = N
    best_branches = []
    best_paths = {}

    def try_set(bset, paths):
        nonlocal best_branches, best_paths
        if len(bset) > len(best_branches):
            best_branches = list(bset)
            best_paths = dict(paths)

    # Phase 1: greedy growing from each adjacent pair
    for u in range(n):
        if time.time() > DEADLINE:
            break
        for v in sorted(ADJ[u]):
            if u >= v:
                continue
            ok, paths = _feasible([u, v], DEADLINE)
            if not ok:
                continue
            current = [u, v]
            current_paths = paths
            grew = True
            while grew:
                grew = False
                if time.time() > DEADLINE:
                    break
                candidates = [w for w in range(n) if w not in current]
                candidates.sort(key=lambda w: -sum(1 for c in current if w in ADJ[c]))
                for w in candidates:
                    if time.time() > DEADLINE:
                        break
                    new_set = current + [w]
                    ok2, paths2 = _feasible(new_set, DEADLINE)
                    if ok2:
                        current = new_set
                        current_paths = paths2
                        grew = True
                        break
            if current is not None and len(current) >= 2:
                try_set(current, current_paths)

    # Phase 2: Bron-Kerbosch over the graph, testing each maximal clique and its extensions
    adj_sets = [ADJ[v] for v in range(n)]
    tested = set()

    def on_clique(clique):
        if time.time() > DEADLINE:
            return
        if len(clique) < 2:
            return
        key = tuple(sorted(clique))
        if key in tested:
            return
        tested.add(key)
        ok, paths = _feasible(clique, DEADLINE)
        if ok:
            try_set(clique, paths)
        # Try to extend by one or two vertices
        base = set(clique)
        for w in range(n):
            if w in base or time.time() > DEADLINE:
                continue
            cand = tuple(sorted(base | {w}))
            if cand in tested:
                continue
            tested.add(cand)
            ok2, paths2 = _feasible(list(cand), DEADLINE)
            if ok2:
                try_set(list(cand), paths2)
                base2 = set(cand)
                for w2 in range(n):
                    if w2 in base2 or time.time() > DEADLINE:
                        continue
                    cand2 = tuple(sorted(base2 | {w2}))
                    if cand2 in tested:
                        continue
                    tested.add(cand2)
                    ok3, paths3 = _feasible(list(cand2), DEADLINE)
                    if ok3:
                        try_set(list(cand2), paths3)

    _bron_kerbosch(adj_sets, set(), set(range(n)), set(), DEADLINE, on_clique)

    # Fallback
    if len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [int(u), int(v)]
        best_paths = {(min(u, v), max(u, v)): [int(u), int(v)]}

    branches_out = [int(b) for b in best_branches]
    paths_out = []
    for i in range(len(branches_out)):
        for j in range(i + 1, len(branches_out)):
            a, b = branches_out[i], branches_out[j]
            key = (min(a, b), max(a, b))
            p = best_paths[key]
            paths_out.append([int(x) for x in p])

    result = {'branches': branches_out, 'paths': paths_out}
    return result
