import random
import time

def construct():
    n = 20
    best_edges = []
    best_count = 0
    
    # Pre-calculate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Number of restarts
    # 120 seconds limit. 
    # Each greedy pass is O(E * n) roughly. E~190, n=20. 3800 ops. Very fast.
    # We can do thousands of restarts.
    
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        random.shuffle(all_edges)
        
        # Adjacency matrix for O(1) lookup
        adj = [[0]*n for _ in range(n)]
        # Common neighbor counts C[u][v]
        # Since n is small, we can recompute C[u][v] on the fly or maintain it.
        # Maintaining C is better.
        # C[u][v] = number of common neighbors.
        # Initially 0.
        
        # To speed up, we can just recompute C for the pairs involved in the check.
        # Or maintain a matrix C.
        C = [[0]*n for _ in range(n)]
        
        edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) is valid
            valid = True
            
            # 1. Check C[u][v]
            if C[u][v] > 2:
                valid = False
            else:
                # 2. Check impact on other pairs
                # For any w, if w is neighbor of u, C[v][w] increases by 1 (because u becomes neighbor of v)
                # Wait, the logic derived earlier:
                # C(u, w) increases by 1 if w in N(v).
                # C(v, w) increases by 1 if w in N(u).
                
                # We need to check all w != u, v
                # Optimization: Only check w that are neighbors of u or v, because only those C values change?
                # No, C(u, w) changes only if w is in N(v).
                # C(v, w) changes only if w is in N(u).
                
                # So we only need to check w in N(u) U N(v).
                
                # Get neighbors of u and v
                # Using adj matrix
                # But adj is 0/1.
                
                # Let's iterate through all w to be safe, n=20 is small.
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # Check C[u][w]
                    # New C[u][w] = C[u][w] + (1 if adj[v][w] else 0)
                    if C[u][w] + adj[v][w] > 2:
                        valid = False
                        break
                    if not valid:
                        break
                    
                    # Check C[v][w]
                    # New C[v][w] = C[v][w] + (1 if adj[u][w] else 0)
                    if C[v][w] + adj[u][w] > 2:
                        valid = False
                        break
            
            if valid:
                # Add edge
                edges.append([u, v])
                adj[u][v] = 1
                adj[v][u] = 1
                
                # Update C matrix
                # C[u][v] is not updated in the matrix for future use? 
                # Actually, we need to update C for all pairs involving u or v.
                # C[x][y] increases if x or y is u or v and the other is a neighbor of the new edge endpoint.
                
                # Let's just recompute C for all pairs? O(n^2) = 400 ops.
                # Or update incrementally.
                # C[u][v] is now |N(u) cap N(v)|. 
                # But we don't store C[u][v] explicitly in a way that is easily updated for all pairs without iterating.
                
                # Given n=20, recomputing C for all pairs is 400 * 20 = 8000 ops?
                # No, C[u][v] = sum(adj[u][k] * adj[v][k] for k in range(n)).
                # Total cost to recompute all C: n^2 * n = 8000 ops.
                # 190 edges * 8000 = 1.5 million ops per restart.
                # 1000 restarts = 1.5 billion ops. Too slow in Python.
                
                # We must update C incrementally.
                
                # When adding (u, v):
                # For all w:
                #   If adj[u][w] is 1, then u is a common neighbor of v and w? No.
                #   u is now a neighbor of v.
                #   So u is in N(v).
                #   If w is in N(u), then u is in N(v) and u is in N(w). So u is a common neighbor of v and w.
                #   So C[v][w] increases by 1 if adj[u][w] == 1.
                #   Similarly, C[u][w] increases by 1 if adj[v][w] == 1.
                
                # So:
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u][w]:
                        C[v][w] += 1
                        C[w][v] += 1
                    if adj[v][w]:
                        C[u][w] += 1
                        C[w][u] += 1
                
                # What about C[u][v]?
                # C[u][v] = |N(u) cap N(v)|.
                # We didn't update C[u][v] in the loop above.
                # But we don't strictly need C[u][v] stored in the matrix if we compute it on the fly for the check?
                # No, the check uses C[u][v].
                # We can compute C[u][v] as sum(adj[u][k]*adj[v][k]) when needed?
                # Or update it.
                # C[u][v] doesn't change when adding (u,v) to other pairs?
                # No, C[u][v] is the count of common neighbors.
                # Adding edge (u,v) does NOT change N(u) or N(v) in a way that affects their intersection with each other?
                # Wait. N(u) gains v. N(v) gains u.
                # Does v become a common neighbor of u and v? No, v is not a neighbor of v (simple graph).
                # Does u become a common neighbor of u and v? No.
                # So C[u][v] remains the same as it was before adding the edge?
                # Let's check.
                # Before: N(u), N(v). Intersection I.
                # After: N(u) U {v}, N(v) U {u}.
                # Intersection: (N(u) U {v}) cap (N(v) U {u})
                # = (N(u) cap N(v)) U (N(u) cap {u}) U ({v} cap N(v)) U ({v} cap {u})
                # N(u) cap {u} is empty (no self loop).
                # {v} cap N(v) is empty.
                # {v} cap {u} is empty.
                # So C[u][v] is unchanged by the addition of the edge (u,v).
                # However, we checked C[u][v] <= 2 before adding.
                # So we don't need to update C[u][v] in the matrix, it stays the same.
                # But wait, if we add another edge (u, w) later, C[u][v] might change?
                # Yes. If we add (u, w), then w is added to N(u).
                # If w is in N(v), then w becomes a common neighbor of u and v.
                # So C[u][v] increases.
                # So we MUST update C[u][v] when adding edges.
                
                # In the update loop above:
                # We updated C[v][w] and C[u][w].
                # Did we update C[u][v]?
                # The pair is (u, v).
                # The logic for updating C[x][y] when adding (a, b) is:
                # If x==a and y==b: No change (as derived).
                # If x==a and y!=b: C[a][y] increases if b in N(y).
                # If x==b and y!=a: C[b][y] increases if a in N(y).
                
                # So C[u][v] is NOT updated in the loop `for w in range(n)` because we skipped `w==u` and `w==v`.
                # And the pairs involving u and v are (u, w) and (v, w).
                # The pair (u, v) itself is not updated.
                # But does C[u][v] change?
                # We established it does NOT change when adding the edge (u, v) itself.
                # So C[u][v] in the matrix is correct.
                
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            # print(f"New best: {best_count} at restart {restarts}")
            
    return best_edges
