"""Bitmask CSP solver for List Coloring with MRV and forward checking."""
# EVOLVE-BLOCK-START
import time

def construct():
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0

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

    def solve_instance(n, edges, lists, deadline):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks
        init_masks = []
        for l in lists:
            mask = 0
            for c in l:
                mask |= (1 << c)
            init_masks.append(mask)
            
        color = [-1] * n
        avail = init_masks[:]
        
        # MRV with tie-breaking by degree
        def backtrack():
            if time.time() > deadline:
                return False, True  # Timeout
            
            # Find unassigned vertex with min available colors (MRV)
            min_size = 5
            min_v = -1
            min_deg = -1
            for v in range(n):
                if color[v] == -1:
                    size = bin(avail[v]).count('1')
                    if size < min_size:
                        min_size = size
                        min_v = v
                        min_deg = len(adj[v])
                        if size == 0:
                            return False, False  # Failure
                        if size == 1:
                            break
                    elif size == min_size and len(adj[v]) > min_deg:
                        min_v = v
                        min_deg = len(adj[v])
            
            if min_v == -1:
                return True, False  # All assigned
            
            # Try each color in the mask for min_v
            mask = avail[min_v]
            while mask:
                c_bit = mask & -mask
                c = c_bit.bit_length() - 1
                mask ^= c_bit
                
                color[min_v] = c
                old_avail_min_v = avail[min_v]
                avail[min_v] = 0
                
                valid = True
                removed_neighbors = []
                for nb in adj[min_v]:
                    if color[nb] == -1:
                        if avail[nb] & c_bit:
                            avail[nb] ^= c_bit
                            removed_neighbors.append(nb)
                            if avail[nb] == 0:
                                valid = False
                                break
                
                if valid:
                    success, timed_out = backtrack()
                    if success:
                        return True, False
                    if timed_out:
                        color[min_v] = -1
                        avail[min_v] = old_avail_min_v
                        for nb in removed_neighbors:
                            avail[nb] |= c_bit
                        return False, True
                else:
                    color[min_v] = -1
                    avail[min_v] = old_avail_min_v
                    for nb in removed_neighbors:
                        avail[nb] |= c_bit
            
            return False, False

        success, timed_out = backtrack()
        
        if success:
            return [int(c) for c in color]
        else:
            return [0] * n

    out = []
    # Allocate time budget per instance: ~25s each, with safety margin
    time_per_instance = 25.0
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Compute remaining time and cap per-instance deadline
        remaining = DEADLINE - time.time()
        if remaining <= 1.0:
            out.append([0] * n)
            continue
        
        instance_deadline = min(time.time() + time_per_instance, DEADLINE - 1.0)
        sol = solve_instance(n, edges, lists, instance_deadline)
        out.append(sol)
    
    return out
# EVOLVE-BLOCK-END
