"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort neighbors by degree for deterministic ordering if needed, 
        # but we will use MRV dynamically.
        # Initial domains are the lists themselves
        domains = [list(l) for l in lists]
        
        # Pre-compute static order by degree for tie-breaking in MRV
        # Higher degree vertices are usually more constrained
        degree = [len(adj[i]) for i in range(n)]
        
        # Backtracking with Forward Checking
        # We use a recursive function. Python recursion limit might be an issue for n=408?
        # n is small (max 408). Recursion depth 408 is fine with default limit (1000).
        
        import sys
        sys.setrecursionlimit(5000)
        
        def backtrack(assigned_count):
            if assigned_count == n:
                return True
            
            # Select unassigned variable with Minimum Remaining Values (MRV)
            # Tie-breaker: Higher degree (more constrained)
            best_var = -1
            min_len = float('inf')
            max_deg = -1
            
            for v in range(n):
                if len(domains[v]) > 0:
                    # If already assigned, domain length is 1? 
                    # No, we maintain domains as available colors.
                    # If a vertex is assigned, its domain is [color].
                    # We need to track assigned status or check if len(domains[v]) == 1 and it's "locked"?
                    # Better approach: maintain a separate 'assigned' array or check if the domain is a singleton 
                    # and we've "processed" it.
                    # Actually, standard forward checking modifies domains of *unassigned* neighbors.
                    # If a vertex is assigned, we shouldn't pick it again.
                    # Let's keep a set of assigned vertices or just check if the domain is a singleton 
                    # and we don't want to re-evaluate it.
                    # Simpler: maintain a list of unassigned vertices.
                    pass
            
            # Let's refine the variable selection.
            # We maintain a list of unassigned vertices.
            # To avoid O(n) scan every time, we could use a heap, but n is small.
            
            # Find unassigned vertex with smallest domain size
            unassigned = [v for v in range(n) if len(domains[v]) > 1 or (len(domains[v]) == 1 and not assigned[v])]
            # Wait, if a vertex is assigned, its domain is [c]. If unassigned, domain is list of availables.
            # If an unassigned vertex has domain size 1, it's forced.
            
            # Actually, if we assign a vertex, we set its domain to [color].
            # We need to know which are assigned.
            
            # Let's restructure to track assigned explicitly.
            return True # Placeholder, logic below

        # Revised Backtracking Logic
        assigned = [False] * n
        colors = [None] * n
        
        def select_variable():
            best_v = -1
            min_size = float('inf')
            max_deg = -1
            for v in range(n):
                if not assigned[v]:
                    size = len(domains[v])
                    if size == 0:
                        return -1 # Failure
                    if size < min_size or (size == min_size and degree[v] > max_deg):
                        min_size = size
                        max_deg = degree[v]
                        best_v = v
            return best_v

        def assign(v, c):
            assigned[v] = True
            colors[v] = c
            domains[v] = [c]
            
            # Forward Checking
            for u in adj[v]:
                if not assigned[u]:
                    if c in domains[u]:
                        domains[u].remove(c)
                        if len(domains[u]) == 0:
                            return False
            return True

        def unassign(v):
            assigned[v] = False
            colors[v] = None
            # Restore domain? We need to keep a copy of the original domain or restore from neighbors.
            # It's easier to copy the entire domains list before assignment if n is small.
            # n <= 408. Copying 408 lists of size <=3 is cheap.
            pass

        # To make unassign easy, we will copy the domains array before each assignment step.
        
        def backtrack():
            # Check for failures
            for v in range(n):
                if not assigned[v] and len(domains[v]) == 0:
                    return False
            
            v = select_variable()
            if v == -1:
                return True # All assigned
            
            # Try each color in domains[v]
            # Order colors by Least Constraining Value (LCV) heuristic?
            # For simplicity, just try in order.
            for c in domains[v]:
                # Save state
                old_domains = [d[:] for d in domains]
                old_assigned = assigned[:]
                
                if assign(v, c):
                    if backtrack():
                        return True
                
                # Restore state
                domains = old_domains
                assigned = old_assigned
            
            return False

        if backtrack():
            return colors
        else:
            return None

    out = []
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10 second margin
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we can't guarantee a valid answer for remaining.
            # But backtracking is very fast for these graphs.
            # We'll just break and return what we have, but the contract says "invalid result... scores 0".
            # So we must ensure we finish.
            # Given n<=408, this loop will take milliseconds.
            pass
            
        n, edges, lists = instance(splits, seed)
        coloring = solve_csp(n, edges, lists)
        
        # Fallback: If backtracking fails (shouldn't happen), try greedy?
        # The theorem guarantees solvability. Backtracking with FC is complete.
        if coloring is None:
            # This should not happen. If it does, return a dummy to avoid crash?
            # No, better to raise or return empty?
            # The contract says "wrong length or type makes the whole result invalid".
            # So we must return a list of lists of ints.
            # If we can't solve, we might return a list of 0s? That would score 0 for that instance.
            # But "invalid result" usually means type/structure error.
            # A valid structure but wrong colors scores 0 for that instance.
            # So returning [0]*n is a valid *structure* that scores 0.
            # But we expect it to work.
            coloring = [0] * n
            
        # Ensure types are Python int
        # The colors are already ints from the lists.
        out.append(coloring)
        
    return out
# EVOLVE-BLOCK-END
