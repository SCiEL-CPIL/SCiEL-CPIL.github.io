import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for the 150s limit
    
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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort neighbors to make iteration deterministic (optional but good for stability)
        for i in range(n):
            adj[i].sort()

        color = [-1] * n
        
        def backtrack(idx):
            # Check deadline
            if time.time() > deadline:
                return False
            
            # Find uncolored vertex with Minimum Remaining Values (MRV)
            # We iterate through all uncolored vertices to find the one with the fewest available colors
            min_count = float('inf')
            min_vertex = -1
            min_avail = []
            
            for v in range(n):
                if color[v] != -1:
                    continue
                
                # Calculate available colors
                taken = {color[u] for u in adj[v] if color[u] != -1}
                avail = [c for c in lists[v] if c not in taken]
                
                if len(avail) < min_count:
                    min_count = len(avail)
                    min_vertex = v
                    min_avail = avail
                    
                    if min_count == 0:
                        # Dead end, backtrack immediately
                        return False
                    if min_count == 1:
                        # Can't get better than 1 (unless 0, handled above)
                        # Breaking early here is an optimization if we find a forced move
                        # But we must check all to ensure we pick the true MRV if ties exist?
                        # Actually, MRV just needs the minimum. If we find 1, we can stop searching for smaller,
                        # but we should probably just pick this one.
                        # However, standard MRV picks the global minimum.
                        # Let's continue to be safe or break if we want speed.
                        # For small n, full scan is fine.
                        pass
            
            if min_vertex == -1:
                # All colored
                return True
            
            # Try colors for min_vertex
            for c in min_avail:
                color[min_vertex] = c
                if backtrack(idx + 1):
                    return True
                color[min_vertex] = -1
            
            return False

        if backtrack(0):
            return color
        else:
            # Fallback: This should not happen for these instances
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        # Ensure types are Python ints
        out.append([int(c) for c in coloring])
        
        # Check deadline between instances
        if time.time() > deadline:
            # If we run out of time, we still need to return a valid structure.
            # Since we solve sequentially, if we fail here, we might have partial results.
            # But the problem says "Any broken rule makes the whole result invalid".
            # We must return a list of 4 lists. If we didn't solve the last ones, 
            # we should probably try to fill them with something or just return what we have?
            # The contract says "Return a list of four lists".
            # If we timeout, we score 0. So we must be fast.
            # With n<=408 and list size 3, backtracking is sub-millisecond typically.
            pass

    return out
