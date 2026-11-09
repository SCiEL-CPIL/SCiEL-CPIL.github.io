import random
import time

def construct():
    n = 15
    deadline = time.time() + 55  # 55 seconds limit
    
    def has_path3(adj, s, t):
        # Check if there is a simple path of length 3 between s and t
        # s - u - v - t
        # u != t, v != s, u != v
        for u in range(n):
            if u == s or u == t:
                continue
            if adj[s][u]:
                for v in range(n):
                    if v == s or v == t or v == u:
                        continue
                    if adj[u][v] and adj[v][t]:
                        return True
        return False

    def is_valid(adj, deg):
        # Check if the graph is valid
        # For every pair of vertices with equal degree, they must NOT have a path of length 3 between them
        for s in range(n):
            for t in range(s + 1, n):
                if deg[s] == deg[t]:
                    if has_path3(adj, s, t):
                        return False
        return True

    best_edges = 0
    best_adj = None
    
    # Start with a valid complete bipartite graph K_{7,8}
    # Vertices 0..6 in partition A, 7..14 in partition B
    # This graph has 7*8 = 56 edges and is valid.
    # Why? In a bipartite graph, any path of length 3 goes from A to B to A to B.
    # So endpoints s and t are in different partitions.
    # Wait, the condition is: "Reject if any simple three-edge path has endpoints of equal degree".
    # In K_{7,8}, all vertices in A have degree 8, all in B have degree 7.
    # Any path of length 3 has endpoints in different partitions (e.g., A-B-A-B or B-A-B-A).
    # So endpoints have different degrees (8 and 7). Thus, no violation.
    
    base_adj = [[0]*n for _ in range(n)]
    base_deg = [0]*n
    for i in range(7):
        for j in range(7, 15):
            base_adj[i][j] = 1
            base_adj[j][i] = 1
            base_deg[i] += 1
            base_deg[j] += 1
    
    # Verify base is valid (should be)
    if is_valid(base_adj, base_deg):
        best_edges = 56
        best_adj = [row[:] for row in base_adj]
    else:
        # Fallback to empty graph if base fails (shouldn't happen)
        best_adj = [[0]*n for _ in range(n)]
        best_edges = 0

    # Local Search: Try to add edges or swap edges to increase count
    # We will perform random restarts of a local search that tries to add edges.
    # If adding an edge is invalid, we might try to remove a "blocking" edge to allow it.
    
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        adj = [row[:] for row in best_adj]
        deg = base_deg[:]
        
        # Randomize the order of potential additions
        # Potential additions are all missing edges
        missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(missing_edges)
        
        # Greedy addition phase
        for u, v in missing_edges:
            if adj[u][v]:
                continue
            # Try to add (u, v)
            adj[u][v] = 1
            adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
            
            if is_valid(adj, deg):
                # Keep it
                pass
            else:
                # Revert
                adj[u][v] = 0
                adj[v][u] = 0
                deg[u] -= 1
                deg[v] -= 1
        
        # Check if we improved
        current_edges = sum(sum(row) for row in adj) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in adj]
            
        # If we hit a high score, we can be more aggressive or stop
        if best_edges >= 60:
             # We are doing well. Let's try to see if we can push higher.
             # The theoretical max is 105 (complete graph).
             # But constraints are tight.
             pass

    # Convert best_adj to list of edges
    if best_adj is None:
        return []
        
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                result.append([i, j])
                
    return result
