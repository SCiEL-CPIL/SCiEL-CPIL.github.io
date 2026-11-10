"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [list(l) for l in lists]
        
        # Check for empty domains initially
        for i in range(n):
            if not domains[i]:
                return None

        # Variable ordering: We will dynamically select the variable with the smallest domain.
        # We use a priority queue or simply scan. Given n <= 408, scanning is O(n^2) total which is fine.
        # But we can be smarter. Let's use a simple selection strategy in the recursion.
        
        # To speed up, we can use a heuristic for the initial order, but dynamic MCV is best.
        # Let's implement a recursive backtracking with forward checking.
        
        # We need to track which vertices are uncoloured.
        uncoloured = set(range(n))
        
        # To make MCV selection fast, we can just iterate over uncoloured and find min len(domains).
        # With n=408, this is 408 checks per node. If depth is 408, that's ~160k operations. Very fast.
        
        def backtrack():
            if not uncoloured:
                return True
            
            # Select variable with Minimum Remaining Values (MRV)
            # Tie-breaker: Highest Degree
            min_size = float('inf')
            selected_v = -1
            for v in uncoloured:
                size = len(domains[v])
                if size == 0:
                    return False
                if size < min_size:
                    min_size = size
                    selected_v = v
                elif size == min_size:
                    if len(adj[v]) > len(adj[selected_v]):
                        selected_v = v
            
            if selected_v == -1:
                return False
            
            # Select values
            # Heuristic: Try values that conflict with the fewest neighbours (LCV)
            # Or just try them in order. Given domain size 3, order might not matter much, 
            # but let's try to be smart.
            values = domains[selected_v]
            
            # Store current state of neighbours' domains to restore
            neighbour_domain_states = []
            
            for val in values:
                # Check if val is still in domain (it should be, but good practice)
                if val not in domains[selected_v]:
                    continue
                
                # Assign val to selected_v
                domains[selected_v] = [val]
                uncoloured.remove(selected_v)
                
                # Forward Checking
                consistent = True
                for u in adj[selected_v]:
                    if u in uncoloured:
                        if val in domains[u]:
                            # Save state
                            neighbour_domain_states.append((u, list(domains[u])))
                            # Remove val
                            domains[u].remove(val)
                            if not domains[u]:
                                consistent = False
                                break
                        else:
                            # If val not in domain, no change needed, but we need to know if we modified it?
                            # No, if val not in domain, we didn't modify it.
                            # But we need to restore only what we modified.
                            # My current logic saves state BEFORE removal.
                            # If val not in domain, we didn't save state.
                            # This is correct.
                            pass
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo changes
                uncoloured.add(selected_v)
                domains[selected_v] = values # Restore original domain? 
                # Wait, domains[selected_v] was set to [val]. 
                # We need to restore it to the original list before this iteration.
                # It's better to copy the domain before modifying.
                
                # Let's restructure the undo logic.
                # Actually, the above loop modifies domains[u].
                # We need to restore them.
                for u, orig_dom in reversed(neighbour_domain_states):
                    domains[u] = orig_dom
                
                # Reset selected_v's domain for the next value in the loop
                # It was [val], we want it back to the original values list
                # But `values` is a reference to `domains[selected_v]` at the start of the loop?
                # No, `values = domains[selected_v]` is a list object.
                # If I do `domains[selected_v] = [val]`, I replace the object.
                # So `values` still points to the old list.
                # So `domains[selected_v] = values` restores it.
                domains[selected_v] = values

            return False

        # We need to be careful with the `values` reference.
        # Let's refine the backtrack function to be cleaner.
        
        def backtrack_clean():
            if not uncoloured:
                return True
            
            # MRV + Degree
            min_size = float('inf')
            selected_v = -1
            for v in uncoloured:
                size = len(domains[v])
                if size == 0:
                    return False
                if size < min_size:
                    min_size = size
                    selected_v = v
                elif size == min_size:
                    if len(adj[v]) > len(adj[selected_v]):
                        selected_v = v
            
            if selected_v == -1:
                return False
            
            # Copy the domain for selected_v to iterate and restore
            orig_domain = list(domains[selected_v])
            uncoloured.remove(selected_v)
            
            # Sort values by LCV (Least Constraining Value)
            # Count how many values each candidate eliminates from neighbours
            # Since domain size is small, we can just try them. 
            # Let's stick to simple order first, if it fails, we can optimize.
            # Actually, let's just iterate.
            
            for val in orig_domain:
                # Assign
                domains[selected_v] = [val]
                
                # Forward Check
                consistent = True
                restored = []
                for u in adj[selected_v]:
                    if u in uncoloured:
                        if val in domains[u]:
                            restored.append((u, list(domains[u])))
                            domains[u].remove(val)
                            if not domains[u]:
                                consistent = False
                                break
                
                if consistent:
                    if backtrack_clean():
                        return True
                
                # Undo
                for u, orig in reversed(restored):
                    domains[u] = orig
                domains[selected_v] = orig_domain
                uncoloured.add(selected_v)
            
            return False

        if backtrack_clean():
            # Extract solution
            solution = [0] * n
            for i in range(n):
                solution[i] = domains[i][0]
            return solution
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, return what we have? 
            # The contract says "An instance whose colouring breaks a list or an edge scores 0".
            # "A wrong length or type makes the whole result invalid."
            # So we must return a valid structure. If we can't solve, we should return a dummy valid structure?
            # No, "raw value is the number of properly coloured instances".
            # If we return an empty list or wrong type, it's invalid.
            # We should return a list of lists of ints.
            # If we fail to find a solution, we can return a list of 0s? 
            # That would likely be invalid (edge conflicts or list conflicts).
            # But it's better than crashing.
            # However, the problem states "all are colourable". So we should find them.
            # If we time out, we just append the current best or a placeholder.
            # Given the speed of the solver, this shouldn't happen.
            pass
            
        n, edges, lists = instance(splits, seed)
        sol = solve_instance(n, edges, lists)
        
        if sol is None:
            # Fallback: return a list of zeros (will likely score 0 for this instance)
            sol = [0] * n
            
        # Ensure types are Python int
        sol = [int(x) for x in sol]
        out.append(sol)

    return out
# EVOLVE-BLOCK-END
