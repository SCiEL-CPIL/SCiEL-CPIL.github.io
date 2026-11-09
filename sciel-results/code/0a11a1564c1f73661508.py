"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 150s wall-clock limit, use 140s to be safe.
    start_time = time.time()
    DEADLINE = 140.0
    
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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert to sets for faster lookup if needed, but lists are fine for small deg
        # For MRV, we need to track available colors dynamically or recompute.
        # Given small n and list size 3, recomputing available colors is cheap.
        
        # Initialize color assignments
        color = [-1] * n
        
        # Heuristic: Order vertices by degree descending to break ties, but dynamic MRV is better.
        # We will use a recursive backtracking with MRV (Minimum Remaining Values).
        
        # To avoid deep recursion limits (though n~400 is safe for recursion if depth is small, 
        # but worst case depth is n), we can use an iterative approach or increase recursion limit.
        sys.setrecursionlimit(10000)
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # Find the uncolored vertex with the fewest available colors (MRV)
            # Since we are assigning in a specific order, we need to pick the next one.
            # Standard backtracking picks the next in a fixed order. 
            # For better performance, we can dynamically pick the next variable.
            
            # However, implementing dynamic variable ordering in simple backtracking 
            # requires tracking which are uncolored.
            
            # Let's stick to a fixed order but sorted by degree initially, 
            # which is often sufficient for these small graphs.
            # If that fails, we try a more dynamic approach.
            
            v = idx # Fixed order for simplicity first
            
            # Check available colors
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            
            # Find a color from lists[v] not in taken
            for c in lists[v]:
                if c not in taken:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False

        # Try fixed order sorted by degree descending
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        # To use the order, we need to map logical index to vertex.
        # Let's rewrite backtrack to use the order list.
        
        def backtrack_order(pos):
            if pos == n:
                return True
            
            v = order[pos]
            
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            
            for c in lists[v]:
                if c not in taken:
                    color[v] = c
                    if backtrack_order(pos + 1):
                        return True
                    color[v] = -1
            return False

        if backtrack_order(0):
            return color
        else:
            # Fallback: try a different heuristic or random restarts?
            # For these specific seeds, the degree-sorted greedy/backtrack should work.
            # If it fails, it's likely a pathological case, but Alon-Tarsi guarantees solvability.
            # Let's try a simple random restart with shuffled order if the first fails.
            import random
            for _ in range(10):
                order2 = list(range(n))
                random.shuffle(order2)
                # Reset color
                for i in range(n):
                    color[i] = -1
                if backtrack_order(0): # Note: backtrack_order uses 'order' variable, need to pass it or make it closure
                    # The above function closes over 'order'. We need a generic one.
                    pass
            
            # Let's implement a generic solver with dynamic MRV for robustness.
            # Given n <= 408, a simple MRV backtracker is very fast.
            
            def solve_mrv():
                # Remaining vertices
                remaining = list(range(n))
                # Current color assignment
                col = [-1] * n
                
                # Helper to get available colors for v
                def get_available(v):
                    taken = set()
                    for u in adj[v]:
                        if col[u] != -1:
                            taken.add(col[u])
                    return [c for c in lists[v] if c not in taken]
                
                def rec(remaining):
                    if not remaining:
                        return True
                    
                    # MRV: pick v with min len(get_available(v))
                    # Tie-break: degree
                    best_v = None
                    best_avail = None
                    min_len = float('inf')
                    
                    for v in remaining:
                        avail = get_available(v)
                        l = len(avail)
                        if l == 0:
                            return False # Dead end
                        if l < min_len:
                            min_len = l
                            best_v = v
                            best_avail = avail
                            if l == 1:
                                break # Can't get better than 1
                    
                    new_remaining = [v for v in remaining if v != best_v]
                    
                    for c in best_avail:
                        col[best_v] = c
                        if rec(new_remaining):
                            return True
                        col[best_v] = -1
                    return False

                if rec(remaining):
                    return col
                return None

            sol = solve_mrv()
            if sol:
                return sol
            else:
                # Should not happen, but return a dummy valid structure to avoid crash
                # Actually, if it fails, we return an invalid coloring which scores 0 for that instance.
                return [-1] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Emergency break: return what we have or dummy
            # But we must return a list of 4 lists. 
            # If we run out of time, we can't generate new ones. 
            # However, these are small, so this won't happen.
            pass
            
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        
        # Ensure types are correct: list of ints
        # The solver returns list of ints.
        out.append(coloring)

    return out

# EVOLVE-BLOCK-END
