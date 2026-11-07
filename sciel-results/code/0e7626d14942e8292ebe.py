"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import deque

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
        """Solve list coloring using backtracking with strong propagation."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Use bitmasks for color sets (palette=5, so 5 bits)
        # color_mask[v] is a 5-bit integer where bit c is set if color c is allowed
        color_mask = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            color_mask[v] = mask
        
        color = [-1] * n  # -1 means unassigned
        
        # Propagation function: given an assignment, propagate constraints
        def propagate(start_v):
            """Propagate from start_v. Returns (success, new_assignments, new_masks)."""
            # We'll modify color and color_mask in place, but track changes for undo
            queue = deque()
            if color[start_v] != -1:
                queue.append(start_v)
            
            assignments = []  # list of (vertex, old_color) for undo
            mask_changes = []  # list of (vertex, old_mask) for undo
            
            while queue:
                v = queue.popleft()
                cv = color[v]
                if cv == -1:
                    continue
                cv_bit = 1 << cv
                for u in adj[v]:
                    if color[u] == -1 and (color_mask[u] & cv_bit):
                        # Remove cv from u's domain
                        old_mask = color_mask[u]
                        color_mask[u] &= ~cv_bit
                        mask_changes.append((u, old_mask))
                        
                        if color_mask[u] == 0:
                            # Conflict
                            return False, assignments, mask_changes
                        if color_mask[u] & (color_mask[u] - 1) == 0:
                            # Only one color left
                            new_c = (color_mask[u] & -color_mask[u]).bit_length() - 1
                            old_c = color[u]
                            color[u] = new_c
                            assignments.append((u, old_c))
                            queue.append(u)
            
            return True, assignments, mask_changes
        
        # Backtracking search
        def backtrack():
            # Find unassigned vertex with smallest domain
            best_v = -1
            best_size = 4  # max is 3, but use 4 as initial
            for v in range(n):
                if color[v] == -1:
                    size = bin(color_mask[v]).count('1')
                    if size < best_size:
                        best_size = size
                        best_v = v
                        if size <= 1:
                            break
            
            if best_v == -1:
                # All assigned
                return True
            
            if best_size == 0:
                return False
            
            v = best_v
            # Try each color in the domain
            mask = color_mask[v]
            while mask:
                bit = mask & -mask
                c = bit.bit_length() - 1
                mask ^= bit
                
                # Check if color c conflicts with assigned neighbors
                conflict = False
                for u in adj[v]:
                    if color[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Assign color c to v
                old_color = color[v]
                old_mask = color_mask[v]
                color[v] = c
                color_mask[v] = bit
                
                # Propagate
                success, assignments, mask_changes = propagate(v)
                
                if success:
                    if backtrack():
                        return True
                
                # Undo
                color[v] = old_color
                color_mask[v] = old_mask
                for (uv, old_cv) in reversed(assignments):
                    color[uv] = old_cv
                for (um, old_m) in reversed(mask_changes):
                    color_mask[um] = old_m
            
            return False
        
        if backtrack():
            return color
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        if coloring is None:
            # Fallback: should not happen per Alon-Tarsi, but be safe
            coloring = [0] * n
        # Ensure all are Python ints
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out

# EVOLVE-BLOCK-END
