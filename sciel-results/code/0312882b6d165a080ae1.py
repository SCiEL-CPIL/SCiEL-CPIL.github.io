"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

    def solve_instance(n, edges, lists):
        """
        Solve a list-coloring instance using backtracking with forward checking.
        Returns a valid coloring (list of ints) or None if no valid coloring found.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert to sets for fast lookup
        adj_set = [set(adj[i]) for i in range(n)]
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos = {v: i for i, v in enumerate(order)}
        
        # We'll assign colors in the order defined by 'order'
        # color[v] = -1 means unassigned
        color = [-1] * n
        
        # For each vertex, we need to try colors from its list
        # We use recursion with backtracking
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Try each color in the list
            for c in lists[v]:
                # Check if c conflicts with any already-colored neighbor
                conflict = False
                for u in adj_set[v]:
                    if color[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Assign color
                color[v] = c
                
                # Recurse
                if backtrack(idx + 1):
                    return True
                
                # Undo
                color[v] = -1
            
            return False
        
        # Try backtracking
        if backtrack(0):
            return color
        
        # If backtracking failed (shouldn't happen for these instances), try a different order
        # Actually, let's try multiple orderings if the first fails
        # But first, let's try a simpler approach: random restarts with different orders
        
        return None

    def solve_with_restarts(n, edges, lists, max_restarts=100, time_limit=10.0):
        """
        Try multiple vertex orderings to find a valid coloring.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        adj_set = [set(adj[i]) for i in range(n)]
        
        start_time = time.time()
        
        # First try the most constrained vertex first (highest degree)
        order1 = sorted(range(n), key=lambda x: -len(adj[x]))
        
        def try_ordering(order):
            color = [-1] * n
            pos = {v: i for i, v in enumerate(order)}
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                v = order[idx]
                for c in lists[v]:
                    conflict = False
                    for u in adj_set[v]:
                        if color[u] == c:
                            conflict = True
                            break
                    if conflict:
                        continue
                    
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
                return False
            
            if backtrack(0):
                return color
            return None
        
        # Try the standard ordering
        result = try_ordering(order1)
        if result is not None:
            return result
        
        # If that fails, try random orderings
        rng = Random(42)
        for _ in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            order = list(range(n))
            rng.shuffle(order)
            result = try_ordering(order)
            if result is not None:
                return result
        
        return None

    out = []
    deadline = time.time() + 100  # Leave margin before 120s limit
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Try to solve
        coloring = solve_with_restarts(n, edges, lists)
        
        if coloring is None:
            # Fallback: greedy (may not be valid, but better than nothing)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            adj_set = [set(adj[i]) for i in range(n)]
            color = [-1] * n
            for v in sorted(range(n), key=lambda x: -len(adj[x])):
                taken = {color[u] for u in adj_set[v] if color[u] != -1}
                free = [c for c in lists[v] if c not in taken]
                color[v] = free[0] if free else lists[v][0]
            coloring = color
        
        # Ensure all values are Python int
        out.append([int(c) for c in coloring])
        
        # Check deadline
        if time.time() > deadline:
            # If we're running out of time, fill remaining with greedy
            for splits2, seed2 in ((20, 701), (40, 702), (60, 703), (100, 704))[len(out):]:
                n2, edges2, lists2 = instance(splits2, seed2)
                adj2 = [[] for _ in range(n2)]
                for u, v in edges2:
                    adj2[u].append(v)
                    adj2[v].append(u)
                adj_set2 = [set(adj2[i]) for i in range(n2)]
                color2 = [-1] * n2
                for v in sorted(range(n2), key=lambda x: -len(adj2[x])):
                    taken = {color2[u] for u in adj_set2[v] if color2[u] != -1}
                    free = [c for c in lists2[v] if c not in taken]
                    color2[v] = free[0] if free else lists2[v][0]
                out.append([int(c) for c in color2])
            break
    
    # Ensure we have exactly 4 results
    while len(out) < 4:
        out.append([])
    
    return out[:4]
# EVOLVE-BLOCK-END
