"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization
    using a learned linear combination of graph-theoretic centrality features.
    
    Features used:
    1. Prize-weighted eigenvector centrality (captures high-value hubs)
    2. Degree centrality (normalized by distance)
    3. Betweenness-like centrality (shortest path usage)
    4. Prize/distance ratio (local attractiveness)
    
    The heuristic is a weighted combination designed to guide ants toward
    high-value nodes that are structurally central in the graph.
    """
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure no negative distances
    distance_matrix = np.maximum(distance_matrix, 0.0)
    
    # Feature 1: Prize-weighted eigenvector centrality
    # Create a weighted adjacency matrix where weight = prize[j] / (d[i,j] + eps)
    eps = 1e-9
    weighted_adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                weighted_adj[i, j] = prizes[j] / (distance_matrix[i, j] + eps)
    
    # Power iteration for eigenvector centrality
    eigvec = np.ones(n) / n
    for _ in range(100):
        new_eigvec = weighted_adj @ eigvec
        norm = np.sum(new_eigvec)
        if norm > 0:
            new_eigvec = new_eigvec / norm
        else:
            break
        if np.allclose(new_eigvec, eigvec, atol=1e-8):
            break
        eigvec = new_eigvec
    
    # Node-level prize-weighted eigenvector centrality
    prize_eig_centrality = eigvec * (prizes + eps)
    
    # Feature 2: Degree centrality (inverse distance sum)
    degree_centrality = np.zeros(n)
    for i in range(n):
        inv_dist_sum = 0.0
        for j in range(n):
            if i != j:
                inv_dist_sum += 1.0 / (distance_matrix[i, j] + eps)
        degree_centrality[i] = inv_dist_sum
    
    # Feature 3: Betweenness-like centrality (number of shortest paths through node)
    # Approximate using all-pairs shortest paths
    # Since we're in a complete graph with Euclidean distances, use simple approximation
    # Count how many pairs (i,j) have node k on their "shortest" path
    betweenness = np.zeros(n)
    # For each pair, find if node k is roughly on the path
    # Use a simple heuristic: node k is "central" if d[i,k] + d[k,j] is close to d[i,j]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                # Check if k is approximately on the shortest path from i to j
                path_through_k = distance_matrix[i, k] + distance_matrix[k, j]
                direct = distance_matrix[i, j]
                # If path through k is not much longer than direct, k contributes
                if path_through_k < direct * 1.1:
                    betweenness[k] += 1.0
    
    # Normalize betweenness
    max_betweenness = np.max(betweenness)
    if max_betweenness > 0:
        betweenness = betweenness / max_betweenness
    
    # Feature 4: Prize/distance ratio (local attractiveness)
    prize_dist_ratio = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                prize_dist_ratio[i, j] = (prizes[j] + eps) / (distance_matrix[i, j] + eps)
    
    # Build edge-wise heuristic from node features
    # For edge (i, j), combine:
    # - Prize of destination j
    # - Eigenvector centrality of j
    # - Degree centrality of j
    # - Betweenness of j
    # - Prize/distance ratio for edge (i,j)
    
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Normalize each feature to [0, 1]
            prize_norm = prizes[j]  # already in [0, 1]
            eig_norm = prize_eig_centrality[j] / (np.max(prize_eig_centrality) + eps)
            deg_norm = degree_centrality[j] / (np.max(degree_centrality) + eps)
            bet_norm = betweenness[j]
            pdr_norm = prize_dist_ratio[i, j] / (np.max(prize_dist_ratio) + eps)
            
            # Weighted linear combination
            w_prize = 0.30
            w_eig = 0.25
            w_deg = 0.15
            w_bet = 0.15
            w_pdr = 0.15
            
            heuristic[i, j] = (
                w_prize * prize_norm +
                w_eig * eig_norm +
                w_deg * deg_norm +
                w_bet * bet_norm +
                w_pdr * pdr_norm
            )
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
