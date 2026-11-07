"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit

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
        domains = [set(l) for l in lists]
        assignment = [None] * n
        
        # MRV + Forward Checking
        def backtrack(unassigned_count):
            if unassigned_count == 0:
                return True
            
            # Time check
            if time.time() > deadline:
                return False

            # Select variable with Minimum Remaining Values (MRV)
            # Tie-breaker: Highest degree
            min_len = float('inf')
            best_var = -1
            best_degree = -1
            
            for i in range(n):
                if assignment[i] is None:
                    len_d = len(domains[i])
                    if len_d < min_len or (len_d == min_len and len(adj[i]) > best_degree):
                        min_len = len_d
                        best_var = i
                        best_degree = len(adj[i])
                        if min_len == 1: # Can't get better than 1
                            break
            
            if best_var == -1:
                return True # No unassigned variables left? Should be caught by unassigned_count

            if min_len == 0:
                return False # Domain wipeout

            # Try each value in the domain of best_var
            # Order values by Least Constraining Value (LCV) heuristic might help, 
            # but for small domains, simple order is often fine. 
            # Let's just iterate.
            for val in list(domains[best_var]):
                # Assign
                assignment[best_var] = val
                domains[best_var].clear()
                
                # Forward Check
                conflict = False
                saved_domains = []
                for neighbor in adj[best_var]:
                    if assignment[neighbor] is None:
                        if val in domains[neighbor]:
                            saved_domains.append((neighbor, val))
                            domains[neighbor].remove(val)
                            if not domains[neighbor]:
                                conflict = True
                                break
                
                if not conflict:
                    if backtrack(unassigned_count - 1):
                        return True
                
                # Undo
                assignment[best_var] = None
                domains[best_var].add(val)
                for neighbor, v in saved_domains:
                    domains[neighbor].add(v)

            return False

        if backtrack(n):
            return assignment
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we must return something valid format-wise, 
            # but likely invalid content. However, the problem says "Any broken rule makes the whole result invalid".
            # We should try to finish. If we can't, we might return the best we have? 
            # No, "wrong length or type makes the whole result invalid".
            # So we must return a list of 4 lists of ints.
            # If we fail to solve, we can't really do much else unless we have a fallback.
            # Given the small size, this shouldn't happen.
            pass

        n, edges, lists = instance(splits, seed)
        solution = solve_csp(n, edges, lists)
        
        if solution is None:
            # Fallback: This shouldn't happen for these instances.
            # If it does, return arbitrary valid types to avoid crash, though score will be 0.
            solution = [0] * n
            
        # Ensure types are Python int
        out.append([int(c) for c in solution])

    return out
# EVOLVE-BLOCK-END
