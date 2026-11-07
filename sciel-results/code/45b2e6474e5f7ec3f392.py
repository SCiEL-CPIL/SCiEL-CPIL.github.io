"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

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

    def greedy_fallback(n, edges, lists):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    def solve_list_coloring(n, edges, lists):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Static degree ordering for tie-breaking or if we switch to static
        # But we use dynamic MRV.
        
        domains = [set(l) for l in lists]
        color = [None] * n
        
        # To speed up MRV selection, we can just scan. n is small.
        
        def backtrack():
            min_size = 5
            best_v = -1
            # Find MRV vertex
            for i in range(n):
                if color[i] is None:
                    size = len(domains[i])
                    if size == 0:
                        return False
                    if size < min_size:
                        min_size = size
                        best_v = i
                        if size == 1:
                            break
            
            if best_v == -1:
                return True
                
            # Try colors
            # Sort colors to ensure determinism, though not strictly required
            for c in sorted(domains[best_v]):
                color[best_v] = c
                valid = True
                changes = []
                
                for nb in adj[best_v]:
                    if color[nb] is not None:
                        if color[nb] == c:
                            valid = False
                            break
                        continue
                    if c in domains[nb]:
                        changes.append(nb)
                        domains[nb].remove(c)
                        if len(domains[nb]) == 0:
                            valid = False
                            break
                
                if valid:
                    if backtrack():
                        return True
                
                # Undo
                for nb in changes:
                    domains[nb].add(c)
                color[best_v] = None
                
            return False

        if backtrack():
            return color
        else:
            return greedy_fallback(n, edges, lists)

    out = []
    start_time = time.time()
    deadline = start_time + 100.0
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback for remaining instances if time runs out
            for rem_splits, rem_seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
                if rem_splits == splits and rem_seed == seed:
                    continue
                # This logic is tricky if we are in the middle. 
                # Simpler: just process current one with greedy if time is tight?
                # Or just let it run. The solver is fast.
                pass
        
        n, edges, lists = instance(splits, seed)
        coloring = solve_list_coloring(n, edges, lists)
        
        # Ensure Python int types
        coloring = [int(c) for c in coloring]
        
        out.append(coloring)
        
    return out
# EVOLVE-BLOCK-END
