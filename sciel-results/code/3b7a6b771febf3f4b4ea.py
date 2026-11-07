"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    _t0 = time.time()
    _DEADLINE = 130.0  # safety margin before 150s kill

    def instance(splits, seed, palette=5):
        from random import Random
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

    def solve_list_coloring(n, edges, lists):
        """Backtracking CSP solver with MRV + forward checking."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Convert to sets for fast lookup
        adj_set = [set(a) for a in adj]
        
        # Domain for each vertex
        domains = [list(lists[i]) for i in range(n)]
        
        # Order vertices by degree (descending) for initial heuristic, but we'll use dynamic MRV
        # Actually, let's use a recursive backtracking with MRV
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(assignments, domains_copy):
            # Find unassigned vertex with minimum domain size (MRV)
            min_size = float('inf')
            min_vertex = -1
            for v in range(n):
                if v not in assignments:
                    if len(domains_copy[v]) < min_size:
                        min_size = len(domains_copy[v])
                        min_vertex = v
                        if min_size == 0:
                            return None
                        if min_size == 1:
                            break
            if min_vertex == -1:
                return assignments
            
            # Try each color in the domain of min_vertex
            for color in domains_copy[min_vertex]:
                new_assignments = dict(assignments)
                new_assignments[min_vertex] = color
                new_domains = [d[:] for d in domains_copy]
                
                # Forward checking
                valid = True
                for neighbor in adj_set[min_vertex]:
                    if neighbor not in new_assignments:
                        if color in new_domains[neighbor]:
                            new_domains[neighbor].remove(color)
                            if len(new_domains[neighbor]) == 0:
                                valid = False
                                break
                if not valid:
                    continue
                
                result = backtrack(new_assignments, new_domains)
                if result is not None:
                    return result
            return None
        
        result = backtrack({}, domains)
        if result is None:
            return None
        return [result[v] for v in range(n)]

    def greedy(n, edges, lists):
        """Greedy fallback."""
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

    def verify(n, edges, lists, colouring):
        """Verify a colouring is valid."""
        if len(colouring) != n:
            return False
        for i, c in enumerate(colouring):
            if not isinstance(c, int) or isinstance(c, bool):
                return False
            if c not in lists[i]:
                return False
        for u, v in edges:
            if colouring[u] == colouring[v]:
                return False
        return True

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > _DEADLINE - 10:
            # Not enough time left, use greedy for remaining
            n, edges, lists = instance(splits, seed)
            out.append(greedy(n, edges, lists))
            continue
        
        n, edges, lists = instance(splits, seed)
        
        # Try backtracking solver
        colouring = solve_list_coloring(n, edges, lists)
        
        if colouring is None or not verify(n, edges, lists, colouring):
            # Fall back to greedy
            colouring = greedy(n, edges, lists)
        
        # Ensure types are correct
        colouring = [int(c) for c in colouring]
        out.append(colouring)
    
    # Final type check
    result = []
    for colouring in out:
        result.append([int(c) for c in colouring])
    return result
# EVOLVE-BLOCK-END
