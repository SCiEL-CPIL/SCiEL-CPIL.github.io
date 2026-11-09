import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']

# Precompute edge indices
edge_index = {}
for i, (u, v) in enumerate(edge_order):
    if u > v:
        u, v = v, u
    edge_index[(u, v)] = i

def get_adjacency(K):
    """
    Build adjacency list for edges with index < K.
    Returns a list of lists of neighbors.
    """
    adj = [[] for _ in range(n)]
    for i in range(K):
        u, v = edge_order[i]
        adj[u].append(v)
        adj[v].append(u)
    # Optional: sort neighbors? Maybe by degree?
    # For now, just return.
    return adj

def find_hamiltonian_cycle(K):
    """
    Find a Hamiltonian cycle in the graph formed by the first K edges.
    Returns the cycle as a list of vertices [v0, v1, ..., v_{n-1}] such that
    (v0, v1), ..., (v_{n-2}, v_{n-1}), (v_{n-1}, v0) are all in the graph.
    Returns None if no such cycle exists.
    """
    adj = get_adjacency(K)
    
    # Quick check: if any vertex has degree 0, return None
    for i in range(n):
        if len(adj[i]) == 0:
            return None
            
    # Backtracking DFS
    # We fix the start vertex to 0 to break symmetry.
    # visited[i] = True if i is in the path
    visited = [False] * n
    path = [0]
    visited[0] = True
    
    # Precompute unvisited count?
    # We can use a recursive function.
    # To avoid recursion depth issues (n=24 is fine), and for speed, we can use iterative or just recursive.
    # Python recursion limit is usually 1000, so 24 is fine.
    
    def dfs(u):
        if len(path) == n:
            # Check if u is connected to 0
            if 0 in adj[u]:
                return True
            return False
        
        # Heuristic: try neighbors with fewer unvisited neighbors first? (Warnsdorff's rule)
        # This often speeds up backtracking significantly.
        neighbors = adj[u]
        # Sort neighbors by the number of unvisited neighbors they have
        # This is a good heuristic for Hamiltonian paths.
        def get_unvisited_count(v):
            count = 0
            for w in adj[v]:
                if not visited[w]:
                    count += 1
            return count
        
        # We need to be careful with the sorting cost. n is small, so it's fine.
        # But we only need to sort if the list is not too small.
        if len(neighbors) > 1:
            neighbors = sorted(neighbors, key=get_unvisited_count)
            
        for v in neighbors:
            if not visited[v]:
                visited[v] = True
                path.append(v)
                if dfs(v):
                    return True
                path.pop()
                visited[v] = False
        return False

    if dfs(0):
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Binary search for the minimum K
    # Lower bound: n (24 edges needed)
    # Upper bound: len(edge_order) (276)
    # But we know the seed score is 614? No, seed score is 614 raw value?
    # Wait, if seed score is 614, and max index is 275, score is 276.
    # Let's re-verify the seed score.
    # Seed cycle: 0-1-2-...-23-0.
    # Edges: (0,1), (1,2), ..., (22,23), (23,0).
    # Let's find the max index of these edges in edge_order.
    # We can just calculate it.
    seed_cycle = list(range(n))
    max_idx = -1
    for i in range(n):
        u = seed_cycle[i]
        v = seed_cycle[(i+1)%n]
        if u > v: u, v = v, u
        idx = edge_index.get((u,v), -1)
        if idx > max_idx: max_idx = idx
    seed_score = max_idx + 1
    
    # The binary search range
    lo = n  # 24
    hi = len(edge_order)  # 276
    
    best_cycle = seed_cycle
    best_score = seed_score
    
    # We want to minimize the score.
    # If we find a cycle with max index < best_score - 1, we update.
    # Actually, we can just binary search for the smallest K.
    
    # However, backtracking might be slow for some K.
    # We can iterate K from lo to hi and stop when we find a cycle?
    # No, binary search is better if the property is monotonic.
    # Property: "Graph with first K edges has a Hamiltonian cycle" is monotonic.
    # If G_K has a HC, then G_{K+1} has a HC.
    
    # But backtracking might time out for some K.
    # We can add a time check inside the binary search steps.
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        cycle = find_hamiltonian_cycle(mid)
        
        if cycle is not None:
            # Found a cycle with max index < mid
            # The score is mid (since max index is mid-1, score is mid)
            # Actually, the score is the largest one-based arrival index.
            # If we use edges with indices < mid, the largest index is at most mid-1.
            # So the score is at most mid.
            # We want to minimize the score.
            # So we update best and try smaller K.
            best_cycle = cycle
            best_score = mid  # This is an upper bound on the true score for this cycle
            # Wait, the actual score of this specific cycle might be less than mid.
            # But we know it's <= mid.
            # To be precise, we should calculate the actual max index of the edges in this cycle.
            actual_max_idx = -1
            for i in range(n):
                u = cycle[i]
                v = cycle[(i+1)%n]
                if u > v: u, v = v, u
                idx = edge_index.get((u,v), -1)
                if idx > actual_max_idx: actual_max_idx = idx
            actual_score = actual_max_idx + 1
            if actual_score < best_score:
                best_score = actual_score
                # We don't need to update best_cycle if we just want the score, 
                # but we need to return a valid cycle.
                # The cycle found is valid.
            
            hi = mid - 1
        else:
            lo = mid + 1

    # If we didn't find a better cycle, return the seed cycle
    # But we should check if the seed cycle is better than what we found?
    # No, we initialized best_cycle to seed_cycle and best_score to seed_score.
    # And we only update if we find a cycle with score < best_score.
    # Wait, in the loop, I set best_score = mid, which might be larger than actual_score.
    # Let's refine the update logic.
    
    # Let's re-run the logic cleanly.
    
    return best_cycle

# EVOLVE-BLOCK-END
