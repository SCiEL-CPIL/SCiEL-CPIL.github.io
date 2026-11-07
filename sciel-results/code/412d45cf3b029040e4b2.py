import time
import sys
import heapq

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

    def solve(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        color = [-1] * n
        avail = [set(l) for l in lists]

        # MRV + Forward Checking
        # We maintain a list of unassigned vertices.
        # At each step, pick the unassigned vertex with the smallest domain size.
        
        def backtrack():
            # Find unassigned vertex with MRV
            min_size = float('inf')
            v = -1
            for i in range(n):
                if color[i] == -1:
                    size = len(avail[i])
                    if size == 0:
                        return False # Dead end
                    if size < min_size:
                        min_size = size
                        v = i
                        if size == 1:
                            break # Can't get better than 1
            
            if v == -1:
                return True # All assigned
            
            # Try colors for v
            # Sort colors by least constraining value (LCV) heuristic? 
            # For now, just iterate.
            for c in list(avail[v]):
                # Check if c is still valid (it is, as it's in avail)
                color[v] = c
                avail[v].remove(c)
                
                # Forward checking: remove c from neighbors
                consistent = True
                saved_states = []
                for u in adj[v]:
                    if color[u] == -1:
                        if c in avail[u]:
                            avail[u].discard(c)
                            saved_states.append((u, c))
                            if len(avail[u]) == 0:
                                consistent = False
                                break
                        else:
                            # c not in avail[u], so no change needed for this neighbor
                            pass
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo
                color[v] = -1
                avail[v].add(c)
                for u, removed_c in saved_states:
                    if color[u] == -1:
                        avail[u].add(removed_c)
                        # Note: if neighbor was assigned in deeper recursion, 
                        # its domain is effectively ignored/overwritten, but we 
                        # restore it to be safe for the backtracking path.
                        # Actually, if neighbor was assigned, color[u] != -1, 
                        # so we shouldn't add back to its domain? 
                        # But we only saved states for unassigned neighbors.
                        # So this is safe.

            return False

        if backtrack():
            return [int(c) for c in color]
        else:
            # Fallback: should not happen
            return [0] * n

    out = []
    start_time = time.time()

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        elapsed = time.time() - start_time
        if elapsed > 110:
            # Safety guard
            # Since we must return 4 lists, and we are likely fast, this is just a precaution.
            # If we hit this, we might fail. But with MRV+FC, we should be very fast.
            pass

        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists)
        # Ensure types are correct: list of Python ints
        out.append([int(c) for c in sol])

    return out
