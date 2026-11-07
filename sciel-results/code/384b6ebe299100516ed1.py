import numpy as np
import time

def heuristics(distance_matrix):
    """
    Computes an edge-attractiveness matrix for ACO TSP.
    Uses a k-NN graph shortest-path metric to penalize long-range/crossing edges.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s kill
    
    n = distance_matrix.shape[0]
    
    # 1. Build a k-Nearest Neighbors graph
    # k=10 is usually enough for connectivity in 2D uniform points
    k = 10
    # Ensure k < n
    if k >= n:
        k = n - 1
        
    # Get indices of k nearest neighbors for each node
    # distance_matrix[i] is the row for node i
    # argsort gives indices sorted by distance
    # We take the first k, excluding self (index 0 in sorted if diagonal is 0)
    
    # Initialize adjacency matrix with infinity for non-edges
    adj = np.full((n, n), np.inf)
    np.fill_diagonal(adj, 0)
    
    for i in range(n):
        # Get sorted indices of distances from i
        # distance_matrix[i] contains distances to all nodes
        # We need to exclude i itself. Since dist[i,i]=0, it will be first.
        # So take indices 1 to k+1
        sorted_indices = np.argsort(distance_matrix[i])
        neighbors = sorted_indices[1:k+1]
        
        for j in neighbors:
            # Symmetrize: add edge if j is in i's kNN
            adj[i, j] = distance_matrix[i, j]
            adj[j, i] = distance_matrix[j, i]
            
    # Check connectivity. If not connected, add more edges (increase k or use MST)
    # For simplicity and speed, if k=10 isn't connected, we can just use the full distance matrix
    # but that defeats the purpose. Let's try to ensure connectivity by adding MST edges if needed.
    # However, for 50 points in unit square, k=10 is almost certainly connected.
    # Let's do a quick BFS check from node 0.
    
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if not visited[v] and adj[u, v] < np.inf:
                visited[v] = True
                queue.append(v)
                
    if not np.all(visited):
        # Graph is disconnected. Fallback to using all edges with a heavy penalty for long ones
        # Or simply increase k until connected
        # For robustness, let's just use the inverse distance if disconnected, 
        # but weight it by local density?
        # Actually, let's just add the missing edges from the global MST to connect components.
        # Simpler: If disconnected, use k=15.
        k = 15
        adj = np.full((n, n), np.inf)
        np.fill_diagonal(adj, 0)
        for i in range(n):
            sorted_indices = np.argsort(distance_matrix[i])
            neighbors = sorted_indices[1:k+1]
            for j in neighbors:
                adj[i, j] = distance_matrix[i, j]
                adj[j, i] = distance_matrix[j, i]
                
        # Re-check connectivity
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if not visited[v] and adj[u, v] < np.inf:
                    visited[v] = True
                    queue.append(v)
                    
        if not np.all(visited):
            # Last resort: use inverse distance
            return 1.0 / (distance_matrix + 1e-9)

    # 2. Compute all-pairs shortest paths on the graph
    # Use Floyd-Warshall for simplicity and speed for N=50
    # O(N^3) = 125,000 ops, very fast.
    dist_graph = adj.copy()
    
    for k_node in range(n):
        # Vectorized Floyd-Warshall step
        # dist_graph[i][j] = min(dist_graph[i][j], dist_graph[i][k_node] + dist_graph[k_node][j])
        # Note: np.inf + x = np.inf
        via_k = dist_graph[:, k_node][:, np.newaxis] + dist_graph[k_node, :]
        dist_graph = np.minimum(dist_graph, via_k)
        
        if time.time() > deadline:
            # If we run out of time, return what we have or a fallback
            # But FW is fast.
            pass

    # 3. Construct Heuristic
    # Attractiveness is inversely proportional to graph distance
    # Add small epsilon to avoid division by zero (diagonal is 0)
    # Diagonal should be 0 or 1? Heuristic for i->i is not used in TSP path construction (no self loops).
    # But we return a matrix.
    
    heuristic = 1.0 / (dist_graph + 1e-9)
    
    # Ensure positive values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Optional: Boost edges that are in a good reference tour?
    # The graph distance already favors short local edges.
    # Let's stick to the graph distance heuristic as per the directive's "non-crossing flow" idea.
    
    return heuristic
