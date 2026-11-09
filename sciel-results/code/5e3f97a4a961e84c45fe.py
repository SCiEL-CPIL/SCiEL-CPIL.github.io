"""Rewritten program using a backtracking CSP solver for List Coloring."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 120s limit, leave 10s margin
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

    def solve_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial available colors as bitmasks
        # Colors are 0..4, so we need 5 bits
        avail = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            avail[v] = mask
        
        color = [-1] * n
        assigned = [False] * n
        
        # Precompute popcount for 5-bit integers (0..31)
        popcount = [bin(i).count('1') for i in range(32)]
        
        # Heuristic: MRV (Minimum Remaining Values)
        
        def backtrack():
            if time.time() > DEADLINE:
                return False, False # Timeout, signal to stop
            
            # Find unassigned vertex with min available colors
            min_size = float('inf')
            min_v = -1
            for v in range(n):
                if not assigned[v]:
                    size = popcount[avail[v]]
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size == 0:
                            return False, True # Failure, not timeout
                        if size == 1:
                            break # Best case for MRV
            
            if min_v == -1:
                return True, True # All assigned
            
            # Try each color for min_v
            # Iterate through possible colors 0..4
            for c in range(5):
                if (avail[min_v] >> c) & 1:
                    # Assign c to min_v
                    color[min_v] = c
                    assigned[min_v] = True
                    
                    # Propagate: remove c from neighbors' avail
                    valid = True
                    removed = [] # neighbors that had c removed
                    for nb in adj[min_v]:
                        if not assigned[nb]:
                            if (avail[nb] >> c) & 1:
                                avail[nb] &= ~(1 << c)
                                removed.append(nb)
                                if avail[nb] == 0:
                                    valid = False
                                    break
                    
                    if valid:
                        success, timed_out = backtrack()
                        if success:
                            return True, True
                        if timed_out:
                            # Restore state before returning timeout
                            assigned[min_v] = False
                            color[min_v] = -1
                            for nb in removed:
                                avail[nb] |= (1 << c)
                            return False, True
                    else:
                        # Restore state for this failed branch
                        assigned[min_v] = False
                        color[min_v] = -1
                        for nb in removed:
                            avail[nb] |= (1 << c)
            
            # All colors failed
            return False, True

        success, timed_out = backtrack()
        
        if success:
            # Ensure types are correct: int
            return [int(c) for c in color]
        else:
            # If it failed or timed out, return a dummy list to maintain structure
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Not enough time for remaining instances, return empty/invalid for them?
            # Better to just stop and return what we have, padding with invalids?
            # The contract says "Return a list of four lists".
            # If we run out of time, we can't generate the rest.
            # We'll break and pad with [0] * n if we know n, but we don't if we didn't generate.
            # So we must generate instance first to know n.
            pass
        
        n, edges, lists = instance(splits, seed)
        
        # Check time again
        if time.time() > DEADLINE:
             out.append([0] * n)
             continue
             
        sol = solve_instance(n, edges, lists)
        out.append(sol)
        
    return out
# EVOLVE-BLOCK-END
