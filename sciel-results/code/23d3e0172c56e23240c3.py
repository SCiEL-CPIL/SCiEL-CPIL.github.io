import itertools

def construct():
    # Generate 37 points in hexagon of radius 3
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if max(abs(q), abs(r), abs(q+r)) <= 3:
                points.append((q, r))
    
    # Identify outer ring
    outer = [p for p in points if max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])) == 3]
    
    # Precompute edges between all 37 points
    # dist_sq(p1, p2) == 1
    def dist_sq(p1, p2):
        q1, r1 = p1
        q2, r2 = p2
        dq = q1 - q2
        dr = r1 - r2
        return dq*dq + dr*dr + dq*dr
        
    # We want to remove 5 points from 'outer'
    best_score = -1
    best_set = None
    
    # Optimization: Precompute adjacency list for the 37 points
    # index map
    idx_map = {p: i for i, p in enumerate(points)}
    adj = {i: [] for i in range(len(points))}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if dist_sq(points[i], points[j]) == 1:
                adj[i].append(j)
                adj[j].append(i)
                
    # For a subset S to remove, the number of edges in the remaining graph is:
    # Total Edges - Edges incident to S + Edges within S
    # Total Edges is constant.
    # We want to minimize (Edges incident to S - Edges within S)
    # Which is equivalent to minimizing the "cut" weight.
    
    total_edges = sum(len(v) for v in adj.values()) // 2
    
    for rem_combo in itertools.combinations(outer, 5):
        rem_indices = [idx_map[p] for p in rem_combo]
        rem_set = set(rem_indices)
        
        # Edges incident to rem_set: sum of degrees of nodes in rem_set
        # But edges between two nodes in rem_set are counted twice in sum of degrees,
        # but should be subtracted only once from total?
        # Let E(G) be total edges.
        # E(G \ S) = E(G) - (Edges with at least one end in S)
        # Edges with at least one end in S = Sum(deg(v) for v in S) - Edges(within S)
        # So E(G \ S) = E(G) - Sum(deg(v) for v in S) + Edges(within S)
        
        deg_sum = sum(len(adj[i]) for i in rem_indices)
        
        # Count edges within rem_set
        edges_within = 0
        for i in range(len(rem_indices)):
            for j in range(i+1, len(rem_indices)):
                if rem_indices[j] in adj[rem_indices[i]]:
                    edges_within += 1
                    
        edges_in_remaining = total_edges - deg_sum + edges_within
        
        if edges_in_remaining > best_score:
            best_score = edges_in_remaining
            best_set = [p for p in points if idx_map[p] not in rem_set]
            
    # Convert to list of lists
    return [list(p) for p in best_set]
