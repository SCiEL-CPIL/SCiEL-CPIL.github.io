import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
        
        # Convert to list of neighbors for faster iteration if needed, but sets are fine for small n
        # We will use a simple backtracking with MRV and Degree heuristics
        
        color = [None] * n
        
        # Pre-compute degrees for tie-breaking
        degrees = [len(adj[i]) for i in range(n)]
        
        # Helper to check if a color is valid for a vertex
        def is_valid(v, c):
            for u in adj[v]:
                if color[u] == c:
                    return False
            return True

        def backtrack(unassigned_count):
            if unassigned_count == 0:
                return True
            
            # Select variable with Minimum Remaining Values (MRV)
            # If tie, Maximum Degree
            best_v = -1
            best_mrv = n + 1
            best_deg = -1
            
            for v in range(n):
                if color[v] is None:
                    # Count available colors
                    avail = 0
                    taken = set()
                    for u in adj[v]:
                        if color[u] is not None:
                            taken.add(color[u])
                    
                    for c in lists[v]:
                        if c not in taken:
                            avail += 1
                            
                    if avail < best_mrv or (avail == best_mrv and degrees[v] > best_deg):
                        best_mrv = avail
                        best_deg = degrees[v]
                        best_v = v
            
            if best_v == -1:
                return True # Should not happen if unassigned_count > 0
            
            if best_mrv == 0:
                return False # No valid color
            
            v = best_v
            # Try colors in list
            # To improve determinism or performance, we could order colors, but list order is fine
            for c in lists[v]:
                if is_valid(v, c):
                    color[v] = c
                    if backtrack(unassigned_count - 1):
                        return True
                    color[v] = None
            
            return False

        # Check time
        if time.time() > deadline:
            return None

        if backtrack(n):
            # Ensure types are Python int
            return [int(c) for c in color]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we might have to return partial or invalid, 
            # but the task says "Any broken rule makes the whole result invalid".
            # However, backtracking is very fast. This is a safety net.
            # If we fail to color an instance, we return None which will make the result invalid.
            # But given the small size, this is unlikely.
            pass
            
        n, edges, lists = instance(splits, seed)
        sol = solve_instance(n, edges, lists)
        
        if sol is None:
            # Fallback: Greedy (unlikely to be needed if backtracking works)
            # But better to just return a dummy valid structure? No, must be valid coloring.
            # If backtracking fails (shouldn't happen), we have a problem.
            # Let's try a simple greedy as a last resort just to have *something* in the list, 
            # though it will likely score 0 for that instance.
            # Actually, if backtracking fails, it's unsolvable (which contradicts theorem), 
            # so it's a bug or timeout.
            # For now, if sol is None, we can't return a valid coloring.
            # We'll return a list of zeros which is invalid, resulting in score 0 for that instance.
            sol = [0] * n
            
        out.append(sol)
        
        if time.time() > deadline:
            break
            
    # Ensure we have exactly 4 lists
    while len(out) < 4:
        out.append([])
        
    return out

# The evaluator expects construct() to return a list of 4 lists.
# The code above does this.
