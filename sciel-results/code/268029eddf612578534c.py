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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert adjacency to lists for faster iteration if needed, but sets are fine for small n
        # We will maintain the available colors for each vertex as a list/set
        # Initial available colors are the lists provided
        avail = [set(l) for l in lists]
        
        # Check if any list is empty initially (should not happen per problem statement, but safe)
        for i in range(n):
            if not avail[i]:
                return None # Should not happen

        # Backtracking solver
        # To improve performance, we can use MRV (Minimum Remaining Values)
        # We need a way to efficiently pick the next variable.
        # For N <= 408, a simple scan is O(N) per step, total O(N^2) which is ~160,000 ops, very fast.
        
        colour = [None] * n
        
        def backtrack():
            # Find uncolored vertex with fewest available colors (MRV)
            min_count = float('inf')
            best_v = -1
            for v in range(n):
                if colour[v] is None:
                    count = len(avail[v])
                    if count < min_count:
                        min_count = count
                        best_v = v
                        if count == 1: # Best possible, break early
                            break
                        if count == 0:
                            return False # Dead end
            
            if best_v == -1:
                return True # All colored
            
            # Try each color in avail[best_v]
            # Order: maybe try colors that appear least in neighbors' avail sets?
            # Simple order is usually sufficient for these sizes.
            for c in avail[best_v]:
                # Assign c to best_v
                colour[best_v] = c
                
                # Check neighbors
                valid = True
                # Store state to backtrack
                # We modify avail[neighbor] by removing c
                # We need to restore if this branch fails
                removed = []
                
                for u in adj[best_v]:
                    if colour[u] is not None:
                        if colour[u] == c:
                            valid = False
                            break
                    else:
                        if c in avail[u]:
                            avail[u].remove(c)
                            removed.append(u)
                            if not avail[u]:
                                valid = False
                                break
                
                if valid:
                    if backtrack():
                        return True
                else:
                    # Restore state if we broke early due to conflict or empty list
                    for u in removed:
                        avail[u].add(c)
                    # If valid was False due to neighbor color conflict, we didn't remove c from that neighbor's avail (since it was colored), 
                    # but we might have removed c from others before hitting the conflict?
                    # In the loop above, if colour[u] == c, we break immediately. 
                    # If c in avail[u], we remove and append.
                    # So 'removed' contains all neighbors we successfully removed c from.
                    # We just add c back to them.
                    
                # Restore state
                colour[best_v] = None
                for u in removed:
                    avail[u].add(c)
                
            return False
            
        if backtrack():
            # Ensure types are int
            return [int(c) for c in colour]
        else:
            return None

    out = []
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback: return whatever we have, though this shouldn't happen
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        if colouring is None:
            # This should not happen as instances are guaranteed colorable
            # If it does, we might try a different strategy or just fail
            # But the prompt says "All are colourable"
            # If backtracking fails, it's a bug in my solver or the instance is hard.
            # Given the size, it should be fine.
            # Let's add a simple greedy fallback just in case, though it's likely unnecessary
            # Actually, if backtracking fails, returning invalid is bad.
            # But MRV backtracking is exact. It only fails if no solution exists.
            # So this case should be empty.
            colouring = [0] * n # Dummy, will be invalid
        out.append(colouring)
        
    return out
# EVOLVE-BLOCK-END
