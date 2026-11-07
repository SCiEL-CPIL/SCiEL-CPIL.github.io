"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin before 90s kill
    n = P['n']
    full_edges = set(_edge_key(u, v) for u, v in P['edges'])
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    class TimeExceeded(Exception):
        pass

    def find_hamiltonian_cycle(adj_cur, start, deadline, rng):
        """Find a Hamiltonian cycle in adj_cur starting at start using Warnsdorff heuristic."""
        n2 = len(adj_cur)
        visited = [False] * n2
        path = [start]
        visited[start] = True
        step_count = [0]

        def dfs():
            step_count[0] += 1
            if step_count[0] % 5000 == 0 and time.time() > deadline:
                raise TimeExceeded()

            if len(path) == n2:
                return start in adj_cur[path[-1]]

            u = path[-1]
            nbrs = list(adj_cur[u])

            # Warnsdorff heuristic: prefer neighbors with fewer unvisited neighbors
            def warnsdorff_score(w):
                if visited[w]:
                    return 1000
                return sum(1 for x in adj_cur[w] if not visited[x])

            nbrs.sort(key=lambda w: (warnsdorff_score(w), rng.random()))

            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    path.append(w)
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
        """Verify cycles are edge-disjoint and valid."""
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
        # Work on a copy of adjacency
        adj_cur = [set(s) for s in adj]
        cycles = []
        ok = True

        for c in range(3):
            if time.time() > deadline:
                ok = False
                break

            # Pick a start vertex - try different strategies
            if c == 0:
                start = 0
            elif c == 1:
                # Pick vertex with max remaining degree
                starts = sorted(range(n), key=lambda v: len(adj_cur[v]), reverse=True)
                start = starts[rng.randrange(min(5, len(starts)))]
            else:
                # For third cycle, pick from high-degree vertices
                starts = sorted(range(n), key=lambda v: len(adj_cur[v]), reverse=True)
                start = starts[rng.randrange(min(3, len(starts)))]

            cyc = find_hamiltonian_cycle(adj_cur, start, deadline, rng)
            if cyc is None:
                ok = False
                break

            cycles.append(cyc)

            # Remove edges of this cycle from adj_cur
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
            # Verify edge-disjointness
            if verify_cycles(cycles):
                # Ensure types: list of lists of int
                res = [[int(v) for v in cyc] for cyc in cycles]
                return res

        # Track best partial result
        if len(cycles) >= 1:
            valid_cycles = []
            used = set()
            for cyc in cycles:
                valid = True
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                    k = _edge_key(a, b)
                    if k not in full_edges or k in used:
                        valid = False
                        break
                    used.add(k)
                if valid:
                    valid_cycles.append(cyc)
                else:
                    break

            if len(valid_cycles) > 0 and (best is None or len(valid_cycles) > len(best)):
                best = [[int(v) for v in cyc] for cyc in valid_cycles]

        # Reset rng for diversity
        rng = random.Random(attempt * 7919 + 42)

    if best is not None:
        return best

    # Fallback: try to find at least one cycle
    adj_cur = [set(s) for s in adj]
    cyc = find_hamiltonian_cycle(adj_cur, 0, deadline, random.Random(999))
    if cyc is not None:
        return [[int(v) for v in cyc]]

    # Last resort: return a simple cycle if we can find one
    for v in range(n):
        if len(adj[v]) >= 2:
            return [[v, 0, 1, v]]

    return [[0, 1, 2]]
# EVOLVE-BLOCK-END
