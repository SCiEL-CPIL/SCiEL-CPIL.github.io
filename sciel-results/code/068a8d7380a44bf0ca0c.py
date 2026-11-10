"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
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

    def solve_csp(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to sets for fast membership
        domains = [set(lst) for lst in lists]
        
        # Order vertices by degree descending (more constrained first)
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        # Map from vertex to its position in the order
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
        
        # We'll use a simple backtracking with forward checking
        # Assign colors in order of decreasing degree
        assignment = [None] * n
        # For each vertex, track which neighbors are assigned and what colors they have
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            
            # Determine available colors
            available = list(domains[v])
            for u in adj[v]:
                if assignment[u] is not None:
                    if assignment[u] in available:
                        available.remove(assignment[u])
            
            if not available:
                return False
            
            # Try each available color
            for c in available:
                assignment[v] = c
                if backtrack(idx + 1):
                    return True
                assignment[v] = None
            
            return False
        
        if backtrack(0):
            return assignment
        return None

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        if time.time() > deadline:
            # Not enough time, return what we have
            break
        
        # Try to solve using CSP backtracking
        result = solve_csp(n, edges, lists)
        
        if result is None:
            # Fallback: should not happen per Alon-Tarsi, but handle gracefully
            # Use a simple greedy as last resort
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            color = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {color[u] for u in adj[v] if color[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                color[v] = free[0] if free else lists[v][0]
            result = color
        
        out.append([int(c) for c in result])
    
    # Ensure we return exactly 4 lists
    while len(out) < 4:
        # This shouldn't happen, but pad with empty if needed
        out.append([])
    
    return out[:4]
# EVOLVE-BLOCK-END
