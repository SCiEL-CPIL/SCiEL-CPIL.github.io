"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initialize available colors for each vertex
        # We will maintain a list of available colors for each vertex
        avail = [set(l) for l in lists]
        
        # Order vertices by degree (descending) as a static heuristic for initial selection
        # But MRV will be dynamic. Let's just use a simple recursive backtracking.
        
        color = [-1] * n
        
        def backtrack():
            # Find uncolored vertex with fewest available colors (MRV)
            min_count = float('inf')
            min_vertex = -1
            for v in range(n):
                if color[v] == -1:
                    if len(avail[v]) < min_count:
                        min_count = len(avail[v])
                        min_vertex = v
                        if min_count == 0:
                            break
            
            if min_vertex == -1:
                return True # All colored
            
            if min_count == 0:
                return False # Dead end
            
            # Try each color for min_vertex
            for c in list(avail[min_vertex]):
                color[min_vertex] = c
                # Remove c from neighbors' available sets
                for u in adj[min_vertex]:
                    if color[u] == -1:
                        avail[u].discard(c)
                
                if backtrack():
                    return True
                
                # Undo
                color[min_vertex] = -1
                for u in adj[min_vertex]:
                    if color[u] == -1:
                        avail[u].add(c)
            
            return False

        if backtrack():
            # Ensure types are Python int
            return [int(c) for c in color]
        else:
            # Should not happen if theorem holds, but fallback to first list color
            return [int(l[0]) for l in lists]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, return what we have so far padded with invalid values?
            # No, the contract says "wrong length or type makes the whole result invalid".
            # We must return a valid structure. If we can't solve, we should try to return
            # something that at least has the right length, even if invalid colorings.
            # But the prompt says "Maximize; all are colourable".
            # Given the small size, backtracking should be very fast.
            # If we time out, we just break and return what we have?
            # Actually, if we break, the length will be wrong.
            # Let's just force a fast solution or raise an error?
            # Better: if time is tight, use a greedy fallback to ensure valid length.
            n, edges, lists = instance(splits, seed)
            # Greedy fallback
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            col = [-1] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {col[u] for u in adj[v] if col[u] != -1}
                free = [c for c in lists[v] if c not in taken]
                col[v] = free[0] if free else lists[v][0]
            out.append(col)
            continue
        
        n, edges, lists = instance(splits, seed)
        res = solve_instance(n, edges, lists)
        out.append(res)
        
    return out
# EVOLVE-BLOCK-END
