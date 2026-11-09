import numpy as np
from scipy.cluster import hierarchy

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Base heuristic: Inverse distance
    base_heuristic = 1.0 / (distance_matrix + eps)
    
    if n < 3:
        np.fill_diagonal(base_heuristic, 0)
        return base_heuristic

    # Hierarchical Clustering Approach
    # This approach biases the heuristic towards edges that stay within
    # spatially coherent clusters, reducing long crossing edges.
    n_clusters = 4
    if n < n_clusters:
        n_clusters = max(2, n // 2)
        
    try:
        # Convert square distance matrix to condensed form for linkage
        condensed_dist = hierarchy.squareform(distance_matrix, checks=False)
        
        # Perform Ward linkage (minimizes within-cluster variance)
        linkage_matrix = hierarchy.linkage(condensed_dist, method='ward')
        
        # Cut the dendrogram to get cluster labels
        labels = hierarchy.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
        
        # Create a boolean mask for edges within the same cluster
        same_cluster_mask = (labels[:, None] == labels[None, :])
        
        # Construct final heuristic
        # Boost intra-cluster edges, penalize inter-cluster edges
        boost_factor = 1.0
        decay_factor = 0.2
        
        heuristic = base_heuristic.copy()
        heuristic[same_cluster_mask] *= boost_factor
        heuristic[~same_cluster_mask] *= decay_factor
        
    except Exception:
        # Fallback to simple inverse distance if clustering fails
        heuristic = base_heuristic.copy()
        
    # Ensure diagonal is 0 (ants do not visit self)
    np.fill_diagonal(heuristic, 0)
    
    return heuristic
