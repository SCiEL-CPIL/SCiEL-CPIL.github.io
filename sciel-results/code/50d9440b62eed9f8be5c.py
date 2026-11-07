"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    # Deadline guard: 150s limit, aim to finish by 140s to be safe
    start_time = time.time()
    deadline = start_time + 140.0

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
        
        # Convert adjacency to sets for faster lookup if needed, but lists are fine for small N
        # We will use a set of neighbours for each vertex for O(1) neighbour checks
        adj_set = [set(neighs) for neighs in adj]

        colour = [None] * n
        
        # 1. Greedy Pass: Assign colours greedily to reduce search space
        # Order by degree descending, then by list size ascending (MRV-ish)
        order = sorted(range(n), key=lambda v: (-len(adj[v]), len(lists[v])))
        
        for v in order:
            if colour[v] is not None:
                continue
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            if free:
                # Pick the first available colour
                colour[v] = free[0]
            else:
                # Greedy failed for this vertex, leave it None for backtracking
                pass

        # 2. Backtracking Pass for unassigned vertices
        # We need to solve for the remaining vertices
        # Identify unassigned vertices
        unassigned = [v for v in range(n) if colour[v] is None]
        
        if not unassigned:
            return colour

        # Use recursive backtracking with MRV heuristic
        # To avoid recursion depth issues (though N is small), we use a stack-based approach or recursion with increased limit
        # N max is 8 + 4*100 = 408. Recursion depth 408 is safe in Python (default 1000).
        
        # Precompute domain sizes for MRV
        def get_available(v):
            if colour[v] is not None:
                return 0
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            return len([c for c in lists[v] if c not in taken])

        # Recursive helper
        def backtrack(count):
            if count == 0:
                return True
            
            # Select unassigned vertex with MRV
            best_v = -1
            min_avail = float('inf')
            for v in unassigned:
                if colour[v] is not None:
                    continue
                avail = get_available(v)
                if avail < min_avail:
                    min_avail = avail
                    best_v = v
                    if avail == 0:
                        break
                    if avail == 1:
                        # Can't get better than 1
                        break
            
            if best_v == -1:
                return True # All assigned
            
            if min_avail == 0:
                return False # Dead end
            
            # Try colours for best_v
            taken = {colour[u] for u in adj[best_v] if colour[u] is not None}
            for c in lists[best_v]:
                if c not in taken:
                    colour[best_v] = c
                    if backtrack(count - 1):
                        return True
                    colour[best_v] = None
            
            return False

        # Execute backtracking
        if not backtrack(len(unassigned)):
            # If backtracking fails (should not happen per Alon-Tarsi), return what we have
            # The problem states all are colourable, so this is a safety net
            pass

        return colour

    out = []
    instances = [(20, 701), (40, 702), (60, 703), (100, 704)]
    
    for splits, seed in instances:
        # Check deadline before starting heavy work
        if time.time() > deadline:
            # Return empty lists or partial results? 
            # "A wrong length or type makes the whole result invalid."
            # If we can't solve, we should return a valid structure even if unscored, 
            # but the prompt says "Maximize". If we time out, we score 0.
            # It is better to return a list of correct length with invalid colours? No, "invalid result".
            # If we run out of time, we should return the best we have. 
            # However, if we haven't even started an instance, we can't return a partial colouring for it.
            # We will just break and return whatever we have so far, padded with empty lists? 
            # No, "wrong length ... makes the whole result invalid".
            # So we must ensure we return 4 lists. If we can't solve one, we can return a list of 0s?
            # If we return a list of 0s, it might violate lists/edges, scoring 0 for that instance, 
            # but the result is VALID (correct types/lengths). This is better than an exception/timeout.
            # So, if we run out of time, fill remaining with [0]*n (which will likely fail validity checks for that instance, scoring 0, but keeping the whole result valid).
            
            # First, get n for the current instance to know the size
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
        
        n, edges, lists = instance(splits, seed)
        
        # Solve
        res = solve_instance(n, edges, lists)
        
        # Ensure types are Python int
        res = [int(c) for c in res]
        
        out.append(res)

    # If we broke early due to deadline, out might have fewer than 4 elements
    # We need to pad with valid-length lists to avoid "wrong length" invalidity
    if len(out) < 4:
        for splits, seed in instances[len(out):]:
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)

    return out
# EVOLVE-BLOCK-END
