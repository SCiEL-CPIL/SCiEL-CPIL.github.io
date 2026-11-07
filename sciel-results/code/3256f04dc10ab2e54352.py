import numpy as np
import time

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix for ACO TSP.
    
    Strategy: Spectral Graph-Theoretic Heuristic.
    1. Construct a weighted graph where weights are 1/distance.
    2. Compute the Laplacian eigenvalues/vectors.
    3. Node centrality is based on the variance of the first non-trivial eigenvector (Fiedler vector).
       Nodes with low variance (stable hubs) get higher centrality.
    4. Compute MST. Edges in MST get a boost.
    5. Edge attractiveness = Centrality[i] * Centrality[j] * (MST_Bonus if in MST else 1) * (1/distance)^beta
       This encourages edges between central hubs and penalizes non-MST long edges.
    """
    start_time = time.time()
    deadline = 330.0  # Safety margin
    
    n = len(distance_matrix)
    
    # 1. Basic distance handling
    # Ensure diagonal is 0 for distance, inf for inverse
    dist = distance_matrix.copy()
    np.fill_diagonal(dist, 0.0)
    
    # Avoid division by zero
    min_dist = np.min(dist[dist > 0])
    if min_dist < 1e-9:
        min_dist = 1e-9
    
    # Inverse distance for weights
    # We use a softened inverse to avoid extreme spikes
    # weight = 1 / (dist + min_dist)
    w = 1.0 / (dist + min_dist)
    np.fill_diagonal(w, 0.0)
    
    # 2. Compute Laplacian and Spectral Properties
    # L = D - A, where A is the weighted adjacency matrix
    # D is the degree matrix (row sums of A)
    A = w
    D = np.sum(A, axis=1)
    L = np.diag(D) - A
    
    # Compute eigenvalues and eigenvectors
    # We only need the first few non-trivial eigenvectors
    try:
        # Symmetric eigenproblem
        evals, evecs = np.linalg.eigh(L)
    except Exception:
        # Fallback if eig fails
        return 1.0 / (dist + min_dist)
    
    # The first eigenvector corresponds to eigenvalue 0 (constant vector).
    # The second eigenvector (Fiedler vector) captures the main partitioning structure.
    # We look at the variance of the Fiedler vector components to determine "centrality".
    # Nodes with values close to the mean of the Fiedler vector are "central" or "stable".
    # Nodes with high variance (far from mean) are "peripheral".
    
    # Get Fiedler vector (index 1)
    fiedler_vec = evecs[:, 1]
    mean_fiedler = np.mean(fiedler_vec)
    # Variance of each node's value relative to the mean
    # Actually, we want centrality. 
    # In spectral clustering, points with similar Fiedler vector values are in the same cluster.
    # For TSP, we want a single tour. 
    # Alternative interpretation: 
    # Use the magnitude of the Fiedler vector. 
    # Or, use the "spectral energy" of each node.
    
    # Let's try a different spectral metric: 
    # The "spectral distance" or "embedding" quality.
    # Actually, a simpler spectral heuristic: 
    # Attractiveness is proportional to the dot product of the node's embedding in the low-dimensional spectral space.
    
    # Let's use the first 2 non-trivial eigenvectors (indices 1 and 2) to create 2D coordinates.
    # Then, edge attractiveness can be based on how well the edge fits a "smooth" path in this space.
    # But simpler: 
    # Centrality = 1 / (1 + std_dev_of_node_in_eigenvectors)
    
    # Let's compute the variance of each node across the first K non-trivial eigenvectors.
    K = 3
    eig_sub = evecs[:, 1:K+1]
    # Center the eigenvectors
    eig_sub_centered = eig_sub - np.mean(eig_sub, axis=0, keepdims=True)
    
    # Variance per node
    node_var = np.var(eig_sub_centered, axis=1)
    
    # Centrality: Lower variance means the node is "typical" or central in the spectral space.
    # Higher variance means it's an outlier/peripheral.
    # We want high attractiveness for central nodes.
    # Centrality = 1 / (node_var + epsilon)
    epsilon = 1e-6
    centrality = 1.0 / (node_var + epsilon)
    
    # Normalize centrality to [0, 1] or similar
    max_c = np.max(centrality)
    if max_c > 0:
        centrality = centrality / max_c
    
    # 3. Compute MST
    # We need to identify MST edges to boost them.
    # Kruskal's algorithm or Prim's.
    # For n=50, we can just sort all edges.
    
    # Create list of edges (i, j, weight)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((dist[i,j], i, j))
    
    # Sort by distance (ascending)
    edges.sort(key=lambda x: x[0])
    
    # Union-Find for Kruskal
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    mst_mask = np.zeros((n, n), dtype=bool)
    mst_count = 0
    for d, i, j in edges:
        if union(i, j):
            mst_mask[i, j] = True
            mst_mask[j, i] = True
            mst_count += 1
            if mst_count == n - 1:
                break
    
    # 4. Construct Heuristic Matrix
    # eta[i, j] = centrality[i] * centrality[j] * mst_boost * (1/d)^beta
    # Let beta = 1.0
    # mst_boost = 2.0 if in MST, 1.0 otherwise
    
    eta = np.zeros((n, n), dtype=np.float64)
    
    # Base attractiveness from inverse distance
    # We already have w = 1/(dist + min_dist)
    # Let's use a softer power
    base_attr = w ** 1.0
    
    # Apply centrality product
    # eta[i,j] = base_attr[i,j] * centrality[i] * centrality[j]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                eta[i, j] = 0.0
            else:
                val = base_attr[i, j] * centrality[i] * centrality[j]
                if mst_mask[i, j]:
                    val *= 2.0
                eta[i, j] = val
    
    # Normalize eta
    # We want the mean of off-diagonal entries to be around 1 or similar scale.
    # The ACO uses pheromone * heuristic. Pheromone starts at 1.
    # If heuristic is very large, it dominates. If very small, pheromone dominates.
    # Let's normalize so that the max is 10 and min is 0.1?
    # Or just normalize to mean 1.
    
    off_diag_mask = ~np.eye(n, dtype=bool)
    eta_vals = eta[off_diag_mask]
    
    if len(eta_vals) > 0:
        mean_eta = np.mean(eta_vals)
        if mean_eta > 0:
            eta = eta / mean_eta
    
    # Ensure no zeros (task says non-positive becomes 1e-9, but let's keep positive)
    # If any value is 0, set to 1e-6
    eta[eta < 1e-6] = 1e-6
    
    # Diagonal should be 0 (or irrelevant, but let's set to 0)
    np.fill_diagonal(eta, 0.0)
    
    # Time check
    if time.time() > deadline - 10:
         pass # We are fast enough for n=50
    
    return eta
