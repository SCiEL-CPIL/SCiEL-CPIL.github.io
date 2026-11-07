"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)

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

    def solve_list_coloring(n, edges, lists):
        """
        Solve list coloring for a bipartite planar graph.
        Uses backtracking with forward checking and MRV heuristic.
        Since the graph is bipartite and planar, and lists have size 3,
        it should be solvable efficiently.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert to sets for faster lookup
        adj_sets = [set(neigh) for neigh in adj]
        
        # We'll use a backtracking solver with MRV (Minimum Remaining Values)
        # and forward checking.
        
        color = [-1] * n
        domains = [set(lst) for lst in lists]
        
        # Count variables
        unassigned = [True] * n
        
        def is_valid(v, c):
            for u in adj[v]:
                if color[u] == c:
                    return False
            return True
        
        def backtrack():
            # MRV: pick unassigned vertex with smallest domain
            min_size = n + 1
            min_v = -1
            for v in range(n):
                if unassigned[v]:
                    if len(domains[v]) < min_size:
                        min_size = len(domains[v])
                        min_v = v
                        if min_size <= 1:
                            break
            
            if min_v == -1:
                return True  # All assigned
            
            # Try each color in domain of min_v
            for c in list(domains[min_v]):
                if is_valid(min_v, c):
                    # Assign
                    color[min_v] = c
                    unassigned[min_v] = False
                    old_domains = []
                    
                    # Forward checking: remove c from neighbors' domains
                    for u in adj[min_v]:
                        if unassigned[u]:
                            if c in domains[u]:
                                old_domains.append((u, c))
                                domains[u].discard(c)
                                if len(domains[u]) == 0:
                                    # Conflict
                                    for uu, cc in old_domains:
                                        domains[uu].add(cc)
                                    color[min_v] = -1
                                    unassigned[min_v] = True
                                    break
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue
                    else:
                        # No conflict, recurse
                        if backtrack():
                            return True
                        # Undo forward checking
                        for uu, cc in old_domains:
                            domains[uu].add(cc)
                    
                    # Undo assignment
                    color[min_v] = -1
                    unassigned[min_v] = True
                    for uu, cc in old_domains:
                        domains[uu].add(cc)
            
            return False
        
        if backtrack():
            # Convert to Python int list
            return [int(c) for c in color]
        else:
            # Fallback: should not happen, but return something
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_list_coloring(n, edges, lists)
        # Ensure all values are Python ints
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
