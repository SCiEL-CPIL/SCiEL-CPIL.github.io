import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    A = adjacency.astype(float)
    
    # 1. Degree
    deg = A.sum(axis=1)
    
    # 2. Mean Neighbor Degree
    sum_neighbor_deg = A @ deg
    mean_neighbor_deg = np.zeros(n, dtype=float)
    mask = deg > 0
    mean_neighbor_deg[mask] = sum_neighbor_deg[mask] / deg[mask]
    
    # 3. Sum of Inverse Neighbor Degrees
    # High value means neighbors are low-degree (good candidates).
    # We want to AVOID picking vertices with high SIND.
    inv_deg = 1.0 / (deg + 1.0)
    sum_inv_neighbor_deg = A @ inv_deg
    
    # 4. Triangle Count
    # High value means vertex is in many triangles.
    # We want to AVOID picking vertices with high TriCount.
    A2 = A @ A
    tri_counts = np.diag(A2) / 2.0 # diag(A2) is number of common neighbors, which is 2*triangles? 
    # Wait. (A^2)_{ii} = sum_k A_{ik} A_{ki} = sum_k A_{ik}^2 = degree.
    # (A^3)_{ii} = sum_k A_{ik} (A^2)_{ki} = sum_k A_{ik} (A^2)_{ik}.
    # (A^2)_{ik} is number of length-2 paths between i and k.
    # If i and k are connected, (A^2)_{ik} = number of common neighbors = number of triangles containing edge (i,k).
    # So (A^3)_{ii} = sum_{k in N(i)} (number of triangles containing (i,k)) = 2 * (number of triangles containing i).
    # So tri_counts = np.diag(A @ A @ A) / 2.0.
    # My previous code did A3 = A2 @ A, then diag. This is correct.
    # But I can optimize: tri_counts = np.sum(A * A2, axis=1) / 2.0?
    # A_{ik} * (A^2)_{ik}. Sum over k.
    # Yes, np.sum(A * (A @ A), axis=1) / 2.0 is faster than full A3 matrix multiplication?
    # A @ A is n x n. A * (A @ A) is n x n element-wise. Sum is n.
    # A @ A @ A is n x n @ n x n.
    # For n=200, both are trivial.
    
    tri_counts = np.sum(A * (A @ A), axis=1) / 2.0

    # Rank Normalization
    # Convert to rank percentiles [0, 1]
    def rank_norm(x):
        # Handle NaNs or infinite values? Deg > 0 mask ensures no div by zero in MND.
        # Isolated vertices have deg 0.
        # We will handle isolated vertices separately or ensure they get max score.
        x = x.astype(float)
        # Use argsort of argsort for ranks
        ranks = np.argsort(np.argsort(x))
        # Normalize to [0, 1]
        if n > 1:
            return ranks / (n - 1)
        else:
            return np.zeros_like(x)
    
    r_deg = rank_norm(deg)
    r_mnd = rank_norm(mean_neighbor_deg)
    r_sind = rank_norm(sum_inv_neighbor_deg)
    r_tri = rank_norm(tri_counts)
    
    # Weights
    # Primary: Low Degree
    # Secondary: High Mean Neighbor Degree (neighbors are hubs/bad candidates)
    # Tertiary: Low Sum Inverse Neighbor Degree (neighbors are not low-degree/good candidates)
    # Quaternary: Low Triangle Count
    
    w_deg = 1.0
    w_mnd = 0.6
    w_sind = 0.4
    w_tri = 0.1
    
    scores = -w_deg * r_deg + w_mnd * r_mnd - w_sind * r_sind - w_tri * r_tri
    
    # Isolated vertices (degree 0) should always be picked first.
    # Their rank_norm(deg) will be 0 (lowest), so -0 = 0.
    # But other features might penalize them?
    # MND is 0. Rank 0. +0.
    # SIND is 0. Rank 0. -0.
    # Tri is 0. Rank 0. -0.
    # So isolated vertices get score 0.
    # Vertices with deg > 0 get negative score from -r_deg.
    # So isolated vertices are naturally at the top.
    
    return scores
