import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    # Convert to float for matrix ops
    A = adjacency.astype(float)
    
    # 1. Degree
    deg = A.sum(axis=1)
    
    # 2. Mean Neighbor Degree
    sum_neighbor_deg = A @ deg
    mean_neighbor_deg = np.zeros(n, dtype=float)
    mask = deg > 0
    mean_neighbor_deg[mask] = sum_neighbor_deg[mask] / deg[mask]
    
    # 3. Sum of Inverse Neighbor Degrees
    inv_deg = 1.0 / (deg + 1.0)
    sum_inv_neighbor_deg = A @ inv_deg
    
    # 4. Triangle Count (approximation of clustering)
    # diag(A^3) / 2
    A2 = A @ A
    A3 = A2 @ A
    tri_counts = np.diag(A3) / 2.0
    
    # Normalize all features to Z-scores
    def zscore(x):
        x = x.astype(float)
        mean = x.mean()
        std = x.std()
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - mean) / std
    
    z_deg = zscore(deg)
    z_mean_nbr = zscore(mean_neighbor_deg)
    z_sum_inv = zscore(sum_inv_neighbor_deg)
    z_tri = zscore(tri_counts)
    
    # Weights
    # -deg: Low degree is good. So -z_deg.
    # +mean_nbr: High neighbor degree is good (blocking hubs). So +z_mean_nbr.
    # -sum_inv: Low sum of inverse (i.e., high neighbor degrees) is good. So -z_sum_inv.
    # -tri: Low triangle count (sparse) is good. So -z_tri.
    
    w_deg = 1.0
    w_mean_nbr = 0.5
    w_sum_inv = 0.5
    w_tri = 0.2
    
    scores = -w_deg * z_deg + w_mean_nbr * z_mean_nbr - w_sum_inv * z_sum_inv - w_tri * z_tri
    
    # Isolated vertices
    scores[~mask] = 1e6
    
    return scores
