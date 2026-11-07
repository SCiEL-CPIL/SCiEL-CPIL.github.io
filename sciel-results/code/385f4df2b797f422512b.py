import random, time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    n = P['n']
    full_edges = set(_edge_key(u, v) for u, v in P['edges'])
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    def find_hamiltonian_cycle(adj_cur, start):
        # randomized DFS with backtracking, returns list of vertices or None
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        # order neighbors by degree (Warnsdorff-like)
        def dfs():
            if len(path) == n2:
                return start in adj_cur[path[-1]]
            u = path[-1]
            nbrs = list(adj_cur[u])
            nbrs.sort(key=lambda w: len(adj_cur[w]))
            random.shuffle(nbrs)
            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    path.append(w)
                    if dfs():
                        return True
                    path.pop()
                    visited[w] = False
            return False
        if dfs():
            return list(path)
        return None

    best = None
    rng = random.Random(12345)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # work on a copy of adjacency
        adj_cur = [set(s) for s in adj]
        cycles = []
        ok = True
        for c in range(3):
            # pick a start vertex with max degree
            starts = list(range(n))
            starts.sort(key=lambda v: len(adj_cur[v]))
            start = starts[rng.randrange(len(starts))]
            cyc = find_hamiltonian_cycle(adj_cur, start)
            if cyc is None:
                ok = False
                break
            cycles.append(cyc)
            # remove edges of this cycle
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                k = _edge_key(a, b)
                if k not in full_edges:
                    ok = False
                    break
                adj_cur[a].discard(b)
                adj_cur[b].discard(a)
            if not ok:
                break
        if ok and len(cycles) == 3:
            # verify edge-disjointness
            used = set()
            valid = True
            for cyc in cycles:
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                    k = _edge_key(a, b)
                    if k in used:
                        valid = False
                        break
                    used.add(k)
                if not valid:
                    break
            if valid:
                # ensure types: list of lists of int
                res = [[int(v) for v in cyc] for cyc in cycles]
                return res
        # if 3 failed, maybe 2 is found in this attempt? We track best 2
        if len(cycles) >= 2 and (best is None or len(cycles) > len(best)):
            used = set()
            valid = True
            for cyc in cycles:
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                    k = _edge_key(a, b)
                    if k in used:
                        valid = False
                        break
                    used.add(k)
                if not valid:
                    break
            if valid:
                best = [[int(v) for v in cyc] for cyc in cycles]
        # reset rng for diversity
        rng = random.Random(attempt * 7919)

    if best is not None:
        return best
    # fallback: single Hamiltonian cycle
    adj_cur = [set(s) for s in adj]
    cyc = find_hamiltonian_cycle(adj_cur, 0)
    if cyc is not None:
        return [[int(v) for v in cyc]]
    return [[int(v) for v in list(range(n))]]
