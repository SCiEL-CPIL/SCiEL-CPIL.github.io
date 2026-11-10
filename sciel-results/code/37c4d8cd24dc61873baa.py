"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def color_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Greedy attempt: Degree order
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        def try_greedy(ord):
            color = [-1] * n
            for v in ord:
                taken = set(color[u] for u in adj[v] if color[u] != -1)
                available = [c for c in lists[v] if c not in taken]
                if not available:
                    return None
                color[v] = available[0]
            return color

        res = try_greedy(order)
        if res is not None:
            return res

        # Randomized greedy restarts
        import random
        rng = random.Random(42)
        for _ in range(50):
            o = order[:]
            rng.shuffle(o)
            res = try_greedy(o)
            if res is not None:
                return res

        # Backtracking with Forward Checking
        domains = [set(lists[v]) for v in range(n)]
        color = [-1] * n
        
        def backtrack(unassigned):
            if not unassigned:
                return True
            
            # MRV: pick unassigned vertex with smallest domain
            v = min(unassigned, key=lambda x: len(domains[x]))
            if not domains[v]:
                return False
            
            for c in list(domains[v]):
                color[v] = c
                changes = []
                valid = True
                for u in adj[v]:
                    if u in unassigned:
                        if c in domains[u]:
                            domains[u].remove(c)
                            changes.append((u, c))
                            if not domains[u]:
                                valid = False
                                break
                
                if valid and backtrack(unassigned - {v}):
                    return True
                
                # Undo changes
                for u, c_removed in changes:
                    domains[u].add(c_removed)
                color[v] = -1
            
            return False

        unassigned = set(range(n))
        if backtrack(unassigned):
            return color
        
        raise Exception("Failed to color instance")

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = color_instance(n, edges, lists)
        # Ensure all values are Python ints
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
