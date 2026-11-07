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

    class TimeExceeded(Exception):
        pass

    def find_hamiltonian_cycle(adj_cur, start, deadline, rng, step_limit=500000):
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        step_count = [0]
        max_depth = [0]

        def check_dead_end():
            # Check if any unvisited vertex has degree < 2 in remaining graph
            # (excluding the current path endpoint which will connect to it)
            for v in range(n2):
                if not visited[v]:
                    # Count available neighbors (unvisited or is the start if path is complete)
                    avail = 0
                    for w in adj_cur[v]:
                        if not visited[w] or (w == start and len(path) == n2 - 1):
                            avail += 1
                    if avail < 2:
                        return True
            return False

        def dfs():
            step_count[0] += 1
            if step_count[0] > step_limit:
                raise TimeExceeded()
            if step_count[0] % 20000 == 0 and time.time() > deadline:
                raise TimeExceeded()
            if len(path) == n2:
                return start in adj_cur[path[-1]]
            u = path[-1]
            nbrs = list(adj_cur[u])
            # Warnsdorff heuristic: prefer neighbors with fewer unvisited neighbors
            def warnsdorff_score(w):
                if visited[w]:
                    return 999
                count = sum(1 for x in adj_cur[w] if not visited[x])
                return count
            nbrs.sort(key=lambda w: (warnsdorff_score(w), rng.random()))
            for w in nbrs:
                if not visited[w]:
                    # Check if taking this edge creates a dead end
                    visited[w] = True
                    path.append(w)
                    if not check_dead_end():
                        if dfs():
                            return True
                    path.pop()
                    visited[w] = False
            return False

        try:
            if dfs():
                return list(path)
        except TimeExceeded:
            return None
        return None

    def verify_cycles(cycles):
        used = set()
        for cyc in cycles:
            if len(cyc) < 3:
                return False
            if len(set(cyc)) != len(cyc):
                return False
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                k = _edge_key(a, b)
                if k not in full_edges:
                    return False
                if k in used:
                    return False
                used.add(k)
        return True

    best = None
    rng = random.Random(12345)
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        adj_cur = [set(s) for s in adj]
        cycles = []
        ok = True

        for c in range(3):
            # Pick start vertex with strategy depending on cycle index
            if c == 0:
                # First cycle: random start
                start = rng.randrange(n)
            else:
                # Subsequent cycles: pick vertex with highest remaining degree
                starts = list(range(n))
                starts.sort(key=lambda v: (-len(adj_cur[v]), v))
                # Pick from top third for diversity
                top_third = starts[:max(1, n // 3)]
                start = top_third[rng.randrange(len(top_third))]

            # Adaptive step limit: more steps for first cycle, less for later
            step_limit = 300000 if c == 0 else 200000
            cyc = find_hamiltonian_cycle(adj_cur, start, deadline, rng, step_limit)
            if cyc is None:
                ok = False
                break
            cycles.append(cyc)
            # Remove edges of this cycle
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

        if ok and len(cycles) == 3 and verify_cycles(cycles):
            res = [[int(v) for v in cyc] for cyc in cycles]
            return res

        # Track best partial result
        if len(cycles) >= 1 and verify_cycles(cycles):
            if best is None or len(cycles) > len(best):
                best = [[int(v) for v in cyc] for cyc in cycles]

        # Reset rng for diversity with varied seeds
        rng = random.Random(attempt * 7919 + 12345 + (attempt % 7) * 1000)

    if best is not None:
        return best

    # Fallback: single Hamiltonian cycle
    adj_cur = [set(s) for s in adj]
    rng_fb = random.Random(42)
    cyc = find_hamiltonian_cycle(adj_cur, 0, deadline, rng_fb, 200000)
    if cyc is not None:
        return [[int(v) for v in cyc]]
    return [[int(v) for v in list(range(n))]]
