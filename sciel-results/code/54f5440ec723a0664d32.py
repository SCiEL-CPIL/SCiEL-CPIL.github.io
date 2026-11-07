"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.cluster.vq import kmeans2


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a Cluster-Aware Heuristic.
    
    This approach identifies spatial clusters of customers and biases the
    ant colony to prefer edges that stay within the same cluster, promoting
    geographically compact routes which typically reduce total distance.
    """
    n = distance_matrix.shape[0]
    num_customers = n - 1
    
    # Base: inverse distance
    # Avoid division by zero for diagonal, though we set diagonal to 0 later
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = 1.0 / distance_matrix
    inv_dist[~np.isfinite(inv_dist)] = 0.0
    
    # If there are customers, perform clustering
    if num_customers > 0:
        # Extract customer coordinates (excluding depot at index 0)
        cust_coords = coordinates[1:]
        
        # Choose number of clusters. 
        # A heuristic choice: sqrt(num_customers) or fixed small number.
        # 50 customers -> sqrt(50) ~ 7. Let's try 5 or 6.
        k = min(5, num_customers)
        
        try:
            # Use kmeans2 for clustering. 
            # minit='++' is better for initialization but kmeans2 might not support it in all scipy versions?
            # Actually kmeans2 uses '++' or 'random'. Let's use '++' if available, else 'random'.
            # scipy.cluster.vq.kmeans2(data, k, minit='random')
            # Let's stick to standard kmeans2
            centroids, labels = kmeans2(cust_coords, k, minit='random', seed=42)
        except Exception:
            # Fallback if clustering fails
            labels = np.zeros(num_customers, dtype=int)
            
        # Map labels back to original indices (1 to n-1)
        # labels[0] corresponds to customer index 1
        # labels[n-2] corresponds to customer index n-1
        cluster_ids = np.zeros(n, dtype=int)
        # Depot (0) gets a special cluster ID, say -1 or 0? 
        # We treat depot separately.
        cluster_ids[1:] = labels + 1 # Shift by 1 so depot (0) is distinct if needed, or just use 0 for depot
        cluster_ids[0] = -1
        
        # Create weight matrix W
        W = np.ones((n, n))
        
        # For each pair (i, j) where i != 0 and j != 0
        # If they are in the same cluster, boost attractiveness.
        # If different, penalize.
        
        # Vectorized approach:
        # cluster_ids is shape (n,)
        # Compare all pairs
        same_cluster = (cluster_ids[:, np.newaxis] == cluster_ids[np.newaxis, :])
        
        # We only want to apply this logic to customer-customer edges
        # Create a mask for customer-customer edges
        cust_mask = np.ones((n, n), dtype=bool)
        cust_mask[0, :] = False # No bias for edges FROM depot
        cust_mask[:, 0] = False # No bias for edges TO depot
        np.fill_diagonal(cust_mask, False)
        
        # Apply weights
        # Same cluster: factor 2.0
        # Different cluster: factor 0.5
        W[cust_mask & same_cluster] = 2.0
        W[cust_mask & ~same_cluster] = 0.5
        
        # Combine
        heuristic = inv_dist * W
        
    else:
        heuristic = inv_dist.copy()
        
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic
# EVOLVE-BLOCK-END
