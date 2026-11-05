"""Erdos-Gallai style decomposition: maximize edge-disjoint cycles, leftover edges as pieces."""
import time

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    from random import Random
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(n, edges):
        # adjacency as set of neighbors, and edge set
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))

        pieces = []
        used = set()  # edges already in a cycle

        # Greedy: repeatedly find a simple cycle in remaining edges, extract it.
        # Use a randomized DFS to find cycles; try multiple passes.
        import random
        rng = random.Random(42)

        def find_cycle(start):
            # DFS to find a simple cycle through start using only unused edges
            # returns list of vertices [v0, v1, ..., vk] where v0==vk not included,
            # edges (v0,v1)...(vk,v0) all unused
            stack = [(start, [start])]
            visited = {start}
            while stack:
                node, path = stack.pop()
                # iterate neighbors in random order
                nbrs = list(adj[node])
                # only unused edges
                nbrs = [x for x in nbrs if (min(node,x), max(node,x)) not in used]
                rng.shuffle(nbrs)
                for nb in nbrs:
                    if nb == start and len(path) >= 3:
                        # found cycle
                        return path
                    if nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
            return None

        progress = True
        while progress:
            progress = False
            # try starting from each vertex with remaining degree
            for s in range(n):
                if time.time() > deadline:
                    break
                if len(adj[s] - {x for x in adj[s] if (min(s,x),max(s,x)) in used}) == 0:
                    continue
                cyc = find_cycle(s)
                if cyc is not None:
                    # mark edges used
                    m = len(cyc)
                    for i in range(m):
                        a, b = cyc[i], cyc[(i+1) % m]
                        used.add((min(a,b), max(a,b)))
                    pieces.append(cyc)
                    progress = True
            if time.time() > deadline:
                break

        # leftover edges
        for (u, v) in edge_set:
            if (u, v) not in used:
                pieces.append([u, v])
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(n, edges)
        results.append(pieces)
        if time.time() > deadline:
            # fill remaining with trivial decomposition
            for sp2 in ((20,0.4,11),(25,0.35,12),(30,0.3,13)):
                if sp2 == spec:
                    continue
                if len(results) == results.index(spec) + 1:
                    pass
            # simpler: just break and pad
            break

    # pad if we ran out of time
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    while len(results) < 3:
        idx = len(results)
        n, p, seed = specs[idx]
        edges = instance(n, p, seed)
        results.append([[list(e)] for e in edges])

    return results
