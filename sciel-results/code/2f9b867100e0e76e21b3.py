"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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

    def solve_instance(n, edges, lists):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert to lists for faster iteration
        adj_list = [list(s) for s in adj]
        
        # Use backtracking with MRV (Minimum Remaining Values)
        # For list coloring, each vertex has a list of allowed colors
        # We want to assign a color from the list to each vertex such that
        # adjacent vertices have different colors
        
        color = [None] * n
        # Track which colors are used by each vertex's neighbors
        # We'll use a simple backtracking approach
        
        # Order vertices by degree (descending) as a heuristic
        # But MRV is more important - we'll use dynamic ordering
        
        def backtrack(pos, uncolored):
            if time.time() > deadline:
                return None
            
            if not uncolored:
                return color[:]
            
            # MRV: pick the uncolored vertex with the fewest available colors
            best_v = -1
            best_count = float('inf')
            best_available = None
            
            for v in uncolored:
                # Count available colors for v
                taken = set()
                for u in adj_list[v]:
                    if color[u] is not None:
                        taken.add(color[u])
                available = [c for c in lists[v] if c not in taken]
                if len(available) < best_count:
                    best_count = len(available)
                    best_v = v
                    best_available = available
                    if best_count == 0:
                        return None
                    if best_count == 1:
                        break
            
            # Try each available color for best_v
            for c in best_available:
                color[best_v] = c
                new_uncolored = [v for v in uncolored if v != best_v]
                result = backtrack(pos + 1, new_uncolored)
                if result is not None:
                    return result
                color[best_v] = None
            
            return None
        
        uncolored = list(range(n))
        result = backtrack(0, uncolored)
        
        if result is not None:
            # Verify the solution
            for v in range(n):
                if color[v] not in lists[v]:
                    return None
                for u in adj_list[v]:
                    if color[u] is not None and color[u] == color[v]:
                        return None
            return [int(c) for c in result]
        return None

    def greedy_fallback(n, edges, lists):
        # Simple greedy as fallback
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        color = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {color[u] for u in adj[v] if color[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            if free:
                color[v] = free[0]
            else:
                color[v] = lists[v][0]
        
        # Verify
        for v in range(n):
            if color[v] not in lists[v]:
                return None
            for u in adj[v]:
                if color[u] == color[v]:
                    return None
        return [int(c) for c in color]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5:
            # Not enough time left, use greedy for remaining
            n, edges, lists = instance(splits, seed)
            result = greedy_fallback(n, edges, lists)
            if result is not None:
                out.append(result)
            else:
                out.append([0] * n)
            continue
        
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists)
        if result is None:
            result = greedy_fallback(n, edges, lists)
        if result is not None:
            out.append(result)
        else:
            out.append([0] * n)
    
    return out
# EVOLVE-BLOCK-END
