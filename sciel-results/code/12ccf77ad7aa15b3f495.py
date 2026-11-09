# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a C4-free graph with no independent set of 11 vertices.
    Uses the Polarity Graph of PG(2,7) and prunes vertices to reduce independence number.
    """
    import random
    import sys
    sys.setrecursionlimit(10000)

    q = 7
    # Generate points of PG(2, q)
    points = []
    
    # Case 1: z != 0. Normalize z = 1.
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
            
    # Case 2: z = 0, y != 0. Normalize y = 1.
    for x in range(q):
        points.append((x, 1, 0))
        
    # Case 3: z = 0, y = 0, x != 0. Normalize x = 1.
    points.append((1, 0, 0))
    
    n = len(points) # 57
    
    # Adjacency matrix
    adj = [[False] * n for _ in range(n)]
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            val = (u[0] * v[2] + u[2] * v[0] - 2 * u[1] * v[1]) % q
            if val == 0:
                adj[i][j] = True
                adj[j][i] = True

    # Function to find an independent set of size k
    def find_independent_set(adj, k):
        n = len(adj)
        candidates = list(range(n))
        current_set = []
        
        def backtrack(idx, current):
            if len(current) == k:
                return current[:]
            if idx >= n:
                return None
            
            # Pruning: if not enough candidates left
            if len(current) + (n - idx) < k:
                return None
            
            # Try including candidates[idx]
            v = candidates[idx]
            # Check if v is independent from current
            is_indep = True
            for u in current:
                if adj[u][v]:
                    is_indep = False
                    break
            
            if is_indep:
                res = backtrack(idx + 1, current + [v])
                if res:
                    return res
            
            # Try excluding candidates[idx]
            res = backtrack(idx + 1, current)
            if res:
                return res
            
            return None
        
        return backtrack(0, current_set)

    # Prune vertices
    # We want to remove vertices to break all independent sets of size 11
    # Heuristic: find one, remove the highest degree vertex in it.
    
    removed = [False] * n
    active = list(range(n))
    
    # Simple degree calculation for active vertices
    def get_degree(v):
        deg = 0
        for u in active:
            if u != v and adj[v][u]:
                deg += 1
        return deg

    # Limit iterations to avoid timeout
    max_iter = 100
    for _ in range(max_iter):
        # Filter active vertices for search
        active_adj = [[adj[u][v] for v in active] for u in active]
        
        # Find independent set of size 11 in active subgraph
        # Map back to original indices if needed
        # For simplicity, run search on active list
        
        # If no active vertices left, break
        if len(active) < 11:
            break
            
        # Use a simpler greedy check first?
        # Just run the backtracker on the active subgraph
        # This might be slow.
        
        # Alternative: Just check if alpha < 11 using a simple bound?
        # No, we need to ensure validity.
        
        # Let's try a random removal strategy if search is too slow?
        # No, let's try the search.
        
        # To speed up, we can limit the search depth or use a timeout?
        # Python doesn't have easy timeout.
        
        # Let's just remove the first 18 vertices if search is too complex?
        # No, let's try to find one set.
        
        found_set = find_independent_set(active_adj, 11)
        
        if not found_set:
            break
        
        # found_set contains indices in 'active' list
        # Map to original indices
        orig_indices = [active[i] for i in found_set]
        
        # Remove the vertex with highest degree in the original graph among these
        max_deg = -1
        victim = -1
        for v in orig_indices:
            deg = get_degree(v)
            if deg > max_deg:
                max_deg = deg
                victim = v
        
        if victim != -1:
            removed[victim] = True
            active.remove(victim)
        else:
            break

    # Construct final graph
    final_vertices = [i for i in range(n) if not removed[i]]
    final_n = len(final_vertices)
    
    # Remap indices to 0..final_n-1
    mapping = {old: new for new, old in enumerate(final_vertices)}
    
    final_edges = []
    for i in range(final_n):
        u = final_vertices[i]
        for j in range(i + 1, final_n):
            v = final_vertices[j]
            if adj[u][v]:
                final_edges.append([i, j])
                
    return {'n': final_n, 'edges': final_edges}
# EVOLVE-BLOCK-END
