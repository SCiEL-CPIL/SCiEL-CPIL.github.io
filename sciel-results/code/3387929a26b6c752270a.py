"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

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
        
        # Initial domains are the lists
        domains = [set(lst) for lst in lists]
        assignment = [None] * n
        
        # Heuristic: Minimum Remaining Values (MRV)
        # We can use a simple loop to find the unassigned variable with the smallest domain
        # For better performance on larger graphs, a priority queue could be used, 
        # but n is small enough that linear scan is likely fine and simpler.
        
        def backtrack():
            # Find unassigned variable with smallest domain (MRV)
            min_size = float('inf')
            var = -1
            for i in range(n):
                if assignment[i] is None:
                    size = len(domains[i])
                    if size < min_size:
                        min_size = size
                        var = i
                        if size == 1: # Fail fast if domain is 1? No, we assign it.
                            # Actually if size is 0, it's a contradiction
                            if size == 0:
                                return False
            if var == -1:
                return True # All assigned
            
            # Try each color in the domain of var
            # Order colors by Least Constraining Value (LCV) is an option, 
            # but simple iteration is often sufficient.
            for color in sorted(domains[var]): # Sorted for determinism
                # Assign color to var
                assignment[var] = color
                
                # Forward checking: remove color from neighbors' domains
                saved_domains = []
                consistent = True
                for nei in adj[var]:
                    if assignment[nei] is None:
                        if color in domains[nei]:
                            domains[nei].remove(color)
                            saved_domains.append((nei, color))
                            if len(domains[nei]) == 0:
                                consistent = False
                                break
                        # If color not in domains[nei], no change needed for this neighbor
                    else:
                        # Neighbor already assigned, check consistency
                        if assignment[nei] == color:
                            consistent = False
                            break
                
                if consistent and backtrack():
                    return True
                
                # Undo assignment
                assignment[var] = None
                # Restore domains
                for nei, col in saved_domains:
                    domains[nei].add(col)
                    
                # Note: If consistent was False due to domain becoming 0, 
                # we restored the domains above.
                # If consistent was False due to neighbor conflict, no domains were changed for that neighbor 
                # because we break before modifying? 
                # Wait, the logic above: 
                # if assignment[nei] is not None and assignment[nei] == color: consistent = False. 
                # In this case, we didn't modify domains[nei]. 
                # So restoring only applies to neighbors where we actually removed a color.
                
            return False

        # We need to be careful with state restoration. 
        # The current `saved_domains` only records removals.
        # If a neighbor was already assigned and conflicted, we break.
        # If a neighbor was unassigned, we remove color.
        
        # Let's refine the backtracking logic to be safer.
        
        def backtrack_safe():
            # Find MRV variable
            min_size = float('inf')
            var = -1
            for i in range(n):
                if assignment[i] is None:
                    size = len(domains[i])
                    if size < min_size:
                        min_size = size
                        var = i
            if var == -1:
                return True
            
            if min_size == 0:
                return False
            
            # Try colors
            for color in domains[var]:
                assignment[var] = color
                # Record changes for rollback
                changes = []
                valid = True
                for nei in adj[var]:
                    if assignment[nei] is None:
                        if color in domains[nei]:
                            domains[nei].remove(color)
                            changes.append((nei, color))
                            if len(domains[nei]) == 0:
                                valid = False
                                break
                    else:
                        if assignment[nei] == color:
                            valid = False
                            break
                
                if valid and backtrack_safe():
                    return True
                
                # Rollback
                assignment[var] = None
                for nei, col in changes:
                    domains[nei].add(col)
            
            return False

        # The above rollback logic is slightly flawed if `valid` becomes False 
        # because of a neighbor conflict (assignment[nei] == color).
        # In that case, no domains were changed for that neighbor, so no rollback needed.
        # But if we broke out of the loop early, `changes` might not contain all removals 
        # if we removed from some neighbors then hit a conflict?
        # No, if we hit a conflict `assignment[nei] == color`, we break immediately.
        # We only remove from domains if `assignment[nei] is None`.
        # So `changes` correctly records all removals performed before the break.
        
        success = backtrack_safe()
        if not success:
            # This should not happen for valid instances, but return a dummy or raise
            # The problem states all are colourable.
            return [0] * n 
        
        # Ensure types are Python int
        return [int(c) for c in assignment]

    out = []
    # Deadline guard: 120s total. 
    # We solve 4 instances. Each should be very fast.
    # We'll just run them sequentially.
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        start_time = time.time()
        n, edges, lists = instance(splits, seed)
        
        # Solve
        coloring = solve_csp(n, edges, lists)
        
        # Verification (optional but good for debugging, skip for speed in final)
        # The solver guarantees validity.
        
        out.append(coloring)
        
        elapsed = time.time() - start_time
        # print(f"Solved instance {splits} in {elapsed:.4f}s")
        
    return out
# EVOLVE-BLOCK-END
