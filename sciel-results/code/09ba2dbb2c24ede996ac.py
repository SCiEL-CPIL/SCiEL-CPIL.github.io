"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys

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

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Degrees for tie-breaking
        degrees = [len(adj[i]) for i in range(n)]

        # Domains as sets for O(1) removal
        domains = [set(lst) for lst in lists]
        assigned = [False] * n
        color = [None] * n

        # Increase recursion limit to handle n=408 safely
        sys.setrecursionlimit(3000)

        def backtrack():
            # MRV + Degree Heuristic
            min_domain_size = float('inf')
            best_vertex = -1
            best_degree = -1
            
            for v in range(n):
                if not assigned[v]:
                    size = len(domains[v])
                    if size < min_domain_size or (size == min_domain_size and degrees[v] > best_degree):
                        min_domain_size = size
                        best_vertex = v
                        best_degree = degrees[v]
            
            if best_vertex == -1:
                return True # Success: all vertices assigned
            
            v = best_vertex
            # Try each color in the domain
            # Sorting colors might help if we had value heuristics, but with size 3, it's negligible.
            for c in list(domains[v]):
                assigned[v] = True
                color[v] = c
                
                # Forward Checking
                consistent = True
                # Track changes for undo
                # Since we only remove one color c from neighbors, we just need to add it back.
                # We need to know WHICH neighbors had c removed.
                changed_neighbors = []
                
                for u in adj[v]:
                    if not assigned[u]:
                        if c in domains[u]:
                            domains[u].remove(c)
                            changed_neighbors.append(u)
                            if not domains[u]: # Empty domain
                                consistent = False
                                break
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo
                for u in changed_neighbors:
                    domains[u].add(c)
                assigned[v] = False
                color[v] = None
            
            return False

        if backtrack():
            return color
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_csp(n, edges, lists)
        
        if coloring is None:
            # Fallback: Greedy coloring (should rarely if ever be needed for these instances)
            adj_set = [set() for _ in range(n)]
            for u, v in edges:
                adj_set[u].add(v)
                adj_set[v].add(u)
            coloring = [None] * n
            # Order by degree descending
            order = sorted(range(n), key=lambda x: -len(adj_set[x]))
            for v in order:
                used = {coloring[u] for u in adj_set[v] if coloring[u] is not None}
                available = [c for c in lists[v] if c not in used]
                if available:
                    coloring[v] = available[0]
                else:
                    # If no color available, pick any (will result in invalid coloring, but avoids crash)
                    # In a valid instance, this shouldn't happen.
                    coloring[v] = lists[v][0] if lists[v] else 0

        # Ensure output types are strictly Python int
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
