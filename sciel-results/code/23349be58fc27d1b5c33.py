"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
    
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    candidate_edges = []
    for u in even_verts:
        for v in odd_verts:
            if (u, v) not in base_edges:
                candidate_edges.append((u, v))
    
    def get_diameter(added):
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added:
            adj[u].append(v)
            adj[v].append(u)
        max_d = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = [s]
            for node in q:
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        max_d = max(max_d, dist[nb])
                        q.append(nb)
            if -1 in dist:
                return float('inf')
        return max_d
    
    def is_valid(added):
        # Check all added edges are cross-partition (ensures triangle-free)
        for u, v in added:
            if (u % 2) == (v % 2):
                return False
        return get_diameter(added) <= 4
    
    def try_remove_edges(added):
        """Aggressively try to remove edges while maintaining validity."""
        improved = True
        while improved:
            improved = False
            for i in range(len(added)):
                if time.time() > deadline:
                    return added
                trial = added[:i] + added[i+1:]
                if is_valid(trial):
                    added = trial
                    improved = True
                    break  # restart scan since list changed
        return added
    
    def try_swaps(added):
        """Try swapping edges: remove one, add another to potentially improve."""
        added_set = set(added)
        improved = True
        while improved:
            improved = False
            for i in range(len(added)):
                if time.time() > deadline:
                    return added
                for j in range(len(candidate_edges)):
                    if time.time() > deadline:
                        return added
                    e = candidate_edges[j]
                    if e in added_set:
                        continue
                    trial = added[:i] + [e] + added[i+1:]
                    if is_valid(trial):
                        added = trial
                        added_set = set(added)
                        improved = True
                        break
                if improved:
                    break
        return added
    
    # Generate multiple initial solutions using different hub strategies
    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Single hub at center
    for hub in [11, 12, 13]:
        if time.time() > deadline:
            break
        initial = []
        for v in odd_verts:
            if (hub, v) not in base_edges:
                initial.append((hub, v))
        if is_valid(initial):
            optimized = try_remove_edges(initial)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_solution = optimized
    
    # Strategy 2: Two hubs
    for hub1, hub2 in [(6, 17), (7, 16), (8, 15), (9, 14), (10, 13), (11, 12)]:
        if time.time() > deadline:
            break
        initial = []
        for v in odd_verts:
            # Assign each odd vertex to the closer hub
            d1 = abs(v - hub1)
            d2 = abs(v - hub2)
            h = hub1 if d1 <= d2 else hub2
            if (h, v) not in base_edges:
                if (h, v) not in initial:
                    initial.append((h, v))
        if is_valid(initial):
            optimized = try_remove_edges(initial)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_solution = optimized
    
    # Strategy 3: Pattern-based - connect every kth vertex to create shortcuts
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
        initial = []
        for i in range(0, n, k):
            # Connect i to i+k if not already connected
            if i + k < n and (i, i+k) not in base_edges:
                if i % 2 != (i+k) % 2:  # Ensure bipartite
                    initial.append((i, i+k))
        if is_valid(initial):
            optimized = try_remove_edges(initial)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_solution = optimized
    
    # Strategy 4: Star from multiple centers
    centers = [4, 8, 12, 16, 20]
    for center in centers:
        if time.time() > deadline:
            break
        initial = []
        # Connect center to vertices within distance 4 (to help with diameter)
        for v in range(n):
            if v == center:
                continue
            if abs(v - center) <= 4 and (center, v) not in base_edges:
                if center % 2 != v % 2:  # Ensure bipartite
                    initial.append((center, v))
        if is_valid(initial):
            optimized = try_remove_edges(initial)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_solution = optimized
    
    # If we have a solution, do more local search
    if best_solution is not None:
        best_solution = try_swaps(best_solution)
        best_solution = try_remove_edges(best_solution)
    
    # Fallback: if no valid solution found, use a known valid one
    if best_solution is None:
        # Use a conservative solution
        best_solution = []
        for u in even_verts:
            for v in odd_verts:
                if (u, v) not in base_edges:
                    best_solution.append((u, v))
    
    return [[int(u), int(v)] for u, v in best_solution]
# EVOLVE-BLOCK-END
