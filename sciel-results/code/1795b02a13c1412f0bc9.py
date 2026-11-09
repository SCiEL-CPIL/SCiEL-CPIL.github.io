"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        # Initial cube faces
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7]
        ]
        edges = set()
        for f in faces:
            for i in range(4):
                a, b = f[i], f[(i + 1) % 4]
                edges.add((min(a, b), max(a, b)))
        
        n = 8
        for _ in range(splits):
            # Pick a random face to subdivide
            face_idx = rng.randrange(len(faces))
            face = faces.pop(face_idx)
            
            # Create 4 new inner vertices
            inner = list(range(n, n + 4))
            n += 4
            
            # Connect inner vertices to the original face vertices
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                # Connect face[i] to inner[i]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                # Connect inner[i] to inner[i+1] (cycle within the new square)
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                
                # The new face is formed by [a, b, inner[i+1], inner[i]]
                # Note: The orientation must be consistent for the face list, 
                # but for edge generation, we just need the adjacency.
                # The original face [a, b, c, d] is replaced by 4 new faces:
                # [a, b, inner[i+1], inner[i]]
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            
            # The inner square is a new face
            faces.append(inner)

        # Generate random lists of 3 colors from 0..4
        lists = []
        for _ in range(n):
            lists.append(sorted(rng.sample(range(palette), 3)))
        
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build adjacency list and bitmask adjacency
        adj = [[] for _ in range(n)]
        adj_mask = [0] * n
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Convert lists to bitmasks
        # Palette is 0..4, so bits 0..4 are relevant
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask
        
        colour = [-1] * n
        uncolored = set(range(n))
        
        # Use a recursive backtracking function
        # To avoid recursion depth issues (though n is small), we could use iterative, 
        # but recursion is fine for n ~ 408.
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV Heuristic: Pick vertex with fewest available colors
            min_count = 5
            target = -1
            best_domain = 0
            
            # Iterate over uncolored vertices
            # Optimization: If we find a vertex with 1 color, we can break early
            for v in uncolored:
                count = domains[v].bit_count()
                if count < min_count:
                    min_count = count
                    target = v
                    best_domain = domains[v]
                    if count == 0:
                        return False
                    if count == 1:
                        break
            
            if target == -1:
                return False

            v = target
            uncolored.remove(v)
            
            # Try each color in the domain
            d = best_domain
            while d:
                # Get lowest set bit (color)
                c_mask = d & (-d)
                c = c_mask.bit_length() - 1
                
                # Assign color
                colour[v] = c
                
                # Forward checking: update neighbors
                failed = False
                # We need to store which neighbors had their domains modified to undo them
                modified_neighbors = []
                
                for u in adj[v]:
                    if u in uncolored:
                        if domains[u] & c_mask:
                            domains[u] ^= c_mask
                            modified_neighbors.append(u)
                            if domains[u] == 0:
                                failed = True
                                break
                
                if not failed and backtrack():
                    return True
                
                # Undo
                colour[v] = -1
                for u in modified_neighbors:
                    domains[u] |= c_mask
                
                # Move to next color
                d ^= c_mask
            
            # Restore vertex to uncolored set if all colors failed
            uncolored.add(v)
            return False

        if backtrack():
            return colour
        else:
            # Should not happen due to Alon-Tarsi theorem for these instances
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we are running out of time, fill remaining with dummy solutions
            # to ensure we return a valid structure (list of 4 lists of ints)
            # though they will score 0.
            while len(out) < 4:
                out.append([0])
            break
        
        n, edges, lists = instance(splits, seed)
        # Ensure lists are lists of ints
        sol = solve_instance(n, edges, lists)
        # Convert to Python ints just in case
        out.append([int(c) for c in sol])
        
    # If we exited early due to deadline, ensure we have 4 entries
    while len(out) < 4:
        out.append([0])
        
    return out
# EVOLVE-BLOCK-END
