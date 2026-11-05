"""Path decompositions via Eulerian trail on a doubled-edge multigraph."""
# EVOLVE-BLOCK-START
import time

def construct():
    _t0 = time.time()
    _DEADLINE = 125.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        # Build adjacency with each edge appearing twice (multigraph).
        # Edge ids: 2*i and 2*i+1 for original edge i.
        n = 0
        for (u, v) in edges:
            if u > n: n = u
            if v > n: n = v
        n += 1
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, 2 * i))
            adj[v].append((u, 2 * i))
            adj[u].append((v, 2 * i + 1))
            adj[v].append((u, 2 * i + 1))
        # Hierholzer (iterative)
        used = [False] * (2 * len(edges))
        start = 0
        for x in range(n):
            if adj[x]:
                start = x
                break
        stack = [(start, None)]
        circuit = []
        while stack:
            v, eid = stack[-1]
            # find an unused incident edge
            found = -1
            for k in range(len(adj[v])):
                if not used[adj[v][k][1]]:
                    found = k
                    break
            if found == -1:
                circuit.append(v)
                stack.pop()
            else:
                u, eid2 = adj[v][found]
                used[eid2] = True
                adj[v].pop(found)
                stack.append((u, eid2))
        circuit.reverse()
        # Group into paths: cut when a vertex repeats (consecutive equal).
        paths = []
        cur = [circuit[0]]
        for i in range(1, len(circuit)):
            if circuit[i] == cur[-1]:
                if len(cur) > 1:
                    paths.append(cur)
                cur = [circuit[i]]
            else:
                cur.append(circuit[i])
        if len(cur) > 1:
            paths.append(cur)
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        if time.time() > _DEADLINE:
            # Fallback: each edge as its own path
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        try:
            p = decompose(edges)
            # Validate quickly; if invalid, fallback
            ok = True
            seen = set()
            for path in p:
                for j in range(len(path) - 1):
                    e = tuple(sorted((path[j], path[j + 1])))
                    if e in seen:
                        ok = False
                        break
                    seen.add(e)
                if not ok:
                    break
            if not ok or len(seen) != len(edges):
                p = [list(e) for e in edges]
            results.append(p)
        except Exception:
            results.append([list(e) for e in edges])
    return results
# EVOLVE-BLOCK-END
