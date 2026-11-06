import heapq
import itertools

def construct():
    n = 20
    # adj[u][v] is 1 if edge exists
    adj = [[0] * n for _ in range(n)]
    edges = []
    # degree of each vertex
    deg = [0] * n
    
    # Pre-compute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # We will use a greedy approach.
    # To make it deterministic and potentially better than random,
    # we will try to add edges in an order that balances degrees.
    # We'll iterate through steps. At each step, we check which edges are valid.
    # We pick the valid edge (u, v) that minimizes (deg[u] + deg[v]).
    # This tends to produce a more regular graph, which is often optimal for extremal problems.
    
    remaining_pairs = set(all_pairs)
    
    for _ in range(len(all_pairs)):
        best_edge = None
        best_score = float('inf')
        best_u, best_v = -1, -1
        
        # Iterate through all remaining pairs to find the best valid one
        # Optimization: We can break early if we find a score of 0? No, degrees start at 0.
        # 400 pairs * 20 steps * 20 (check cost) = 160,000 ops. Fast.
        
        for u, v in remaining_pairs:
            # Check if adding (u, v) is valid
            # Condition: For all x in N(v) \ {u}, codeg(u, x) < 2 
            # AND For all y in N(u) \ {v}, codeg(v, y) < 2
            
            valid = True
            
            # Check neighbors of v
            for x in range(n):
                if x == u or x == v:
                    continue
                if adj[v][x]:
                    # Count common neighbors of u and x
                    # Common neighbors y: adj[u][y] and adj[x][y]
                    # Note: v is a neighbor of u (after add) and x (already).
                    # So v is a common neighbor.
                    # We need total common neighbors <= 2.
                    # So we need count of OTHER common neighbors <= 1.
                    count = 0
                    for y in range(n):
                        if y == u or y == v or y == x:
                            continue
                        if adj[u][y] and adj[x][y]:
                            count += 1
                    # Total = count + 1 (for v)
                    if count + 1 > 2:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Check neighbors of u
            for y in range(n):
                if y == u or y == v:
                    continue
                if adj[u][y]:
                    # Count common neighbors of v and y
                    count = 0
                    for z in range(n):
                        if z == u or z == v or z == y:
                            continue
                        if adj[v][z] and adj[y][z]:
                            count += 1
                    # Total = count + 1 (for u)
                    if count + 1 > 2:
                        valid = False
                        break
                        
            if valid:
                score = deg[u] + deg[v]
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
                    best_u, best_v = u, v
                    # Can't get better than 0, but degrees increase.
                    # If we find a low score, we still need to check all to be sure?
                    # Yes, because a later pair might have lower degree sum.
                    
        if best_edge is None:
            # No more edges can be added
            break
            
        # Add the edge
        u, v = best_u, best_v
        adj[u][v] = 1
        adj[v][u] = 1
        deg[u] += 1
        deg[v] += 1
        edges.append([u, v])
        remaining_pairs.remove(best_edge)
        
    return edges
