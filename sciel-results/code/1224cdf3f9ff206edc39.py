import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    all_edge_set = set()
    for a, b in edges:
        all_edge_set.add((min(a,b), max(a,b)))

    deadline = time.time() + 50.0
    used = set()
    cycles = []

    def find_cycle():
        # returns a Hamiltonian cycle (list of n distinct vertices) using only unused edges, or None
        # path[0] = 0 fixed
        path = [0]
        visited = [False]*n
        visited[0] = True
        # available edges = all_edge_set - used
        # We'll track available as a set of canonical edges not in used
        avail = all_edge_set - used

        def edge_avail(a, b):
            return (min(a,b), max(a,b)) in avail

        def dfs():
            if time.time() > deadline:
                return None
            cur = path[-1]
            if len(path) == n:
                # need closing edge cur -> 0
                if edge_avail(cur, 0):
                    return list(path)
                return None
            # candidate next vertices: neighbors of cur, not visited, edge available
            cands = []
            for nb in adj[cur]:
                if not visited[nb] and edge_avail(cur, nb):
                    cands.append(nb)
            if not cands:
                return None
            # heuristic: order by Warnsdorff (fewest onward options)
            def score(v):
                # count onward available edges from v to unvisited vertices
                c = 0
                for nb in adj[v]:
                    if not visited[nb] and edge_avail(v, nb):
                        c += 1
                # also consider edge to 0 if it's the only remaining unvisited? keep simple
                return c
            cands.sort(key=score)
            for nb in cands:
                # forward check before committing
                # tentatively mark nb visited, consume edge (cur,nb)
                visited[nb] = True
                e = (min(cur,nb), max(cur,nb))
                avail.discard(e)
                path.append(nb)
                # forward check: every unvisited vertex must have >=2 available edges to (unvisited ∪ {cur, 0})
                ok = True
                rem = n - len(path)
                if rem > 0:
                    for v in range(n):
                        if visited[v]:
                            continue
                        cnt = 0
                        for nb2 in adj[v]:
                            if (not visited[nb2] or nb2 == cur or nb2 == 0) and edge_avail(v, nb2):
                                cnt += 1
                        if cnt < 2:
                            ok = False
                            break
                    # start 0 must be able to close eventually: needs >=1 available edge to (unvisited ∪ {cur})
                    if ok:
                        cnt0 = 0
                        for nb2 in adj[0]:
                            if (not visited[nb2] or nb2 == cur) and edge_avail(0, nb2):
                                cnt0 += 1
                        if cnt0 < 1:
                            ok = False
                if ok:
                    res = dfs()
                    if res is not None:
                        return res
                # undo
                path.pop()
                avail.add(e)
                visited[nb] = False
            return None
        return dfs()

    for _ in range(3):
        if time.time() > deadline:
            break
        c = find_cycle()
        if c is None:
            break
        # record edges
        ok = True
        for i in range(n):
            a = c[i]
            b = c[(i+1) % n]
            e = (min(a,b), max(a,b))
            if e in used:
                ok = False
                break
        if not ok:
            break
        for i in range(n):
            a = c[i]
            b = c[(i+1) % n]
            used.add((min(a,b), max(a,b)))
        cycles.append([int(v) for v in c])

    if not cycles:
        # fallback: the seed single cycle
        cycles = [[int(v) for v in range(n)]]
    return cycles
