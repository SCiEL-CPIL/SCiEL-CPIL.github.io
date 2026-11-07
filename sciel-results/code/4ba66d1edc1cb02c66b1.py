"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    def color_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Degrees for heuristic tie-breaking
        degrees = [len(neigh) for neigh in adj]
        
        # Convert lists to bitmasks
        # Color c is represented by bit c (1 << c)
        masks = [0] * n
        for i, l in enumerate(lists):
            m = 0
            for c in l:
                m |= (1 << c)
            masks[i] = m
        
        # Check if any domain is empty initially
        for i in range(n):
            if masks[i] == 0:
                return None

        assignment = [-1] * n
        
        # Recursive backtracking with MRV and Degree tie-breaker
        # Using bitmasks for fast domain operations
        
        def backtrack():
            # Find MRV vertex among unassigned
            min_size = 10  # Max domain size is 5
            best_v = -1
            best_deg = -1
            
            for v in range(n):
                if assignment[v] != -1:
                    continue
                size = bin(masks[v]).count('1')
                if size == 0:
                    return False
                # MRV: smallest size
                # Tie-breaker: highest degree
                if size < min_size or (size == min_size and degrees[v] > best_deg):
                    min_size = size
                    best_v = v
                    best_deg = degrees[v]
            
            if best_v == -1:
                # All assigned
                return True
            
            # Save state
            old_mask = masks[best_v]
            
            # Try each color in the mask
            m = old_mask
            while m:
                # Get lowest set bit
                c_bit = m & -m
                c = c_bit.bit_length() - 1
                m ^= c_bit
                
                # Assign color c
                assignment[best_v] = c
                
                # Forward checking: update neighbors' masks
                valid = True
                saved_neighbors = []
                
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        # If neighbor u has color c available, remove it
                        if masks[u] & c_bit:
                            saved_neighbors.append((u, masks[u]))
                            masks[u] ^= c_bit
                            if masks[u] == 0:
                                valid = False
                                break
                
                if valid:
                    if backtrack():
                        return True
                
                # Restore neighbors
                for u, old_m in saved_neighbors:
                    masks[u] = old_m
            
            # Restore best_v to unassigned
            assignment[best_v] = -1
            # masks[best_v] is already old_mask
            
            return False

        if backtrack():
            # Convert to Python ints
            return [int(c) for c in assignment]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = color_instance(n, edges, lists)
        if coloring is None:
            # Should not happen per problem statement, but handle gracefully
            # Return a dummy list to maintain structure, though it will score 0
            coloring = [0] * n
        # Ensure types are Python int
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
