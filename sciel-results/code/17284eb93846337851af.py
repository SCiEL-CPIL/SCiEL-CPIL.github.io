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

    def solve_list_coloring(n, edges, lists):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert to list for faster iteration
        adj_list = [list(s) for s in adj]
        
        colour = [None] * n
        # Track which colors are used by each vertex's neighbors
        # For backtracking, we'll use a simple recursive approach with MRV
        
        # Precompute: for each vertex, its available colors (initially its list)
        # We'll do forward checking
        
        def backtrack(pos):
            if pos == n:
                return True
            
            # MRV: find uncolored vertex with fewest available colors
            best_v = -1
            best_count = 10
            for v in range(n):
                if colour[v] is not None:
                    continue
                # Count available colors
                taken = set()
                for u in adj_list[v]:
                    if colour[u] is not None:
                        taken.add(colour[u])
                avail = 0
                for c in lists[v]:
                    if c not in taken:
                        avail += 1
                if avail < best_count:
                    best_count = avail
                    best_v = v
                    if avail == 0:
                        return False
                    if avail == 1:
                        break
            
            if best_v == -1:
                # All colored
                return True
            
            # Try each available color for best_v
            taken = set()
            for u in adj_list[best_v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            
            for c in lists[best_v]:
                if c in taken:
                    continue
                colour[best_v] = c
                if backtrack(pos + 1):
                    return True
                colour[best_v] = None
            
            return False
        
        # Use iterative approach to avoid recursion depth issues for larger graphs
        # For n up to ~408, recursion should be fine, but let's use a stack-based approach
        
        # Actually, let's use a proper backtracking with stack
        # Order vertices by degree descending as initial ordering, then use MRV dynamically
        
        # Let's try a simpler approach: recursive with sys.setrecursionlimit
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, n * 4 + 100))
        
        result = backtrack(0)
        
        sys.setrecursionlimit(old_limit)
        
        if result:
            return colour
        return None

    def greedy(n, edges, palette):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v]}
            free = [c for c in palette(v) if c not in taken]
            colour[v] = free[0] if free else palette(v)[0]
        return colour

    out = []
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback to greedy for remaining
            n, edges, lists = instance(splits, seed)
            out.append(greedy(n, edges, lambda v: lists[v]))
            continue
        
        n, edges, lists = instance(splits, seed)
        
        # Try backtracking solver
        colour = solve_list_coloring(n, edges, lists)
        
        if colour is None:
            # Fallback to greedy
            colour = greedy(n, edges, lambda v: lists[v])
        
        # Ensure all values are Python int
        out.append([int(c) for c in colour])
    
    return out
# EVOLVE-BLOCK-END
