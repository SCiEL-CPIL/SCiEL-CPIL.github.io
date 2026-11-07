import sys
import time

def construct():
    from random import Random
    
    # Increase recursion limit to be safe for n ~ 400
    sys.setrecursionlimit(5000)

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
        
        # Pre-compute degrees for tie-breaking in MRV
        degrees = [len(adj[i]) for i in range(n)]
        
        # Domains: list of lists (mutable) to allow removal
        # We use lists of available colours. Initially lists[i]
        domains = [list(l) for l in lists]
        assigned = [False] * n
        assignment = [None] * n
        
        # Start time for deadline guard
        start_time = time.time()
        deadline = start_time + 110.0 # 110s limit out of 120s
        
        # Helper to check if we should abort
        def check_deadline():
            return time.time() > deadline

        def backtrack(depth):
            if check_deadline():
                return False
            
            if depth == n:
                return True
            
            # MRV Heuristic: Find unassigned vertex with smallest domain
            min_size = float('inf')
            best_v = -1
            best_deg = -1
            for v in range(n):
                if not assigned[v]:
                    size = len(domains[v])
                    if size < min_size or (size == min_size and degrees[v] > best_deg):
                        min_size = size
                        best_v = v
                        best_deg = degrees[v]
            
            if best_v == -1:
                return True # All assigned
            
            # If any domain is empty, we are in a dead end
            if min_size == 0:
                return False
            
            v = best_v
            # Try each colour in the domain
            # We iterate over a copy because we might modify domains
            for c in domains[v][:]:
                if check_deadline():
                    return False
                
                # Assign c to v
                assigned[v] = True
                assignment[v] = c
                
                # Forward checking: remove c from neighbors' domains
                removed = [] # List of (neighbor, colour) to restore
                conflict = False
                for u in adj[v]:
                    if not assigned[u]:
                        if c in domains[u]:
                            domains[u].remove(c)
                            removed.append((u, c))
                            if len(domains[u]) == 0:
                                conflict = True
                                break
                
                if conflict:
                    # Undo changes
                    for u, col in removed:
                        domains[u].append(col)
                    assigned[v] = False
                    assignment[v] = None
                    continue
                
                # Recurse
                if backtrack(depth + 1):
                    return True
                
                # Undo changes
                for u, col in removed:
                    domains[u].append(col)
                assigned[v] = False
                assignment[v] = None
            
            return False

        # Run solver
        if backtrack(0):
            # Convert to plain Python ints and ensure list type
            return [int(assignment[i]) for i in range(n)]
        else:
            # If solver fails (should not happen for these instances), return empty or dummy
            # But task says all are colourable. If we fail, we score 0 for this instance.
            # Returning a list of wrong length/type makes whole result invalid.
            # So we must return a valid structure even if wrong? 
            # "An instance whose colouring breaks a list or an edge scores 0 for that instance"
            # "wrong length or type makes the whole result invalid"
            # So we must return a list of n ints. If we can't find a valid one, we return an invalid one?
            # No, we must return *something* that is a list of n ints. 
            # If we return a list of 0s, it might break edges, scoring 0 for that instance.
            # It will not make the *whole result* invalid, as long as it's a list of ints.
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        out.append(coloring)
    return out
