import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    # Candidate edges: only between even and odd to ensure triangle-free
    # (u, v) is a candidate if u != v, (u,v) not in base_edges, and u%2 != v%2
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2):
                if (u, v) not in base_edges:
                    candidates.append((u, v))
    
    def compute_diameter(added_edges):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        max_diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            # Check if connected (should be, as base path is there)
            if -1 in dist:
                return n + 1 # Disconnected
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam

    def is_valid(added_edges):
        # Since we only add bipartite edges, triangle-free is guaranteed.
        # Connectivity is guaranteed by the base path.
        # We only need to check diameter.
        return compute_diameter(added_edges) <= 4

    # Deadline guard
    start_time = time.time()
    deadline = 80.0 # 20s margin
    
    best_solution = None
    best_count = float('inf')

    # Greedy Construction
    # Start with no added edges
    current_added = []
    
    # We want to reduce diameter. 
    # Heuristic: Add edges that connect "far" vertices.
    # To be more systematic, we can iterate through candidates and add one if it reduces the diameter significantly.
    
    # A better initial guess: 
    # The path is 0-1-...-23.
    # Diameter is 23.
    # We need diameter <= 4.
    # This implies the graph must be quite dense in terms of short paths.
    
    # Let's try a specific structured approach first to get a valid solution quickly.
    # Connect center vertices to others.
    # Centers are 11, 12.
    # Try adding edges from 11 to all even vertices (0,2,4,6,8,10,14,16,18,20,22)
    # and from 12 to all odd vertices (1,3,5,7,9,13,15,17,19,21,23)
    # This creates a "double star" like structure.
    
    init_edges = []
    # 11 is odd. Connect to evens.
    for v in range(0, n, 2):
        if v != 11: # 11 is odd, so v is even, v!=11 is always true for even v? No, 11 is odd.
            if (11, v) not in base_edges:
                init_edges.append((11, v))
    # 12 is even. Connect to odds.
    for v in range(1, n, 2):
        if v != 12:
            if (12, v) not in base_edges:
                init_edges.append((12, v))
                
    # Check if this initial set is valid
    if is_valid(init_edges):
        best_solution = [list(e) for e in init_edges]
        best_count = len(best_solution)
        
    # Now perform local search / greedy removal
    if best_solution is not None:
        current = best_solution[:]
        removed = True
        while removed and (time.time() - start_time) < deadline:
            removed = False
            # Try removing each edge
            for i in range(len(current)):
                if (time.time() - start_time) > deadline:
                    break
                trial = current[:i] + current[i+1:]
                if is_valid(trial):
                    current = trial
                    removed = True
                    best_count = len(current)
                    best_solution = current[:]
                    break # Restart loop with new shorter list
                    
    # If no valid solution was found (unlikely), try greedy addition from empty
    if best_solution is None:
        current = []
        # Greedy addition
        # While diameter > 4, add the edge that most reduces diameter
        while (time.time() - start_time) < deadline:
            diam = compute_diameter(current)
            if diam <= 4:
                break
            # Find best edge to add
            best_edge = None
            best_diam = diam
            # To save time, we can just pick a random candidate or iterate
            # Given N=24, iterating all candidates is fine
            for e in candidates:
                # Skip if already added
                if e in current: continue
                trial = current + [e]
                d = compute_diameter(trial)
                if d < best_diam:
                    best_diam = d
                    best_edge = e
            if best_edge is None:
                # No single edge reduces diameter? Should not happen if graph is connected
                # Just break or add arbitrary
                break
            current.append(best_edge)
            
        if is_valid(current):
            best_solution = [list(e) for e in current]
            best_count = len(best_solution)
            
            # Greedy removal
            removed = True
            while removed and (time.time() - start_time) < deadline:
                removed = False
                for i in range(len(current)):
                    if (time.time() - start_time) > deadline:
                        break
                    trial = current[:i] + current[i+1:]
                    if is_valid(trial):
                        current = trial
                        removed = True
                        best_count = len(current)
                        best_solution = current[:]
                        break

    if best_solution is None:
        return []

    # Ensure types are correct
    return [[int(u), int(v)] for u, v in best_solution]
