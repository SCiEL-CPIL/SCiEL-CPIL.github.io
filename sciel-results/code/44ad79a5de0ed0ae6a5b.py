"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    # Explicit deadline guard (though heuristic computation is fast)
    start_time = time.time()
    deadline = 290.0  # Safety margin for 300s limit
    
    n = distance_matrix.shape[0]
    if n < 3:
        return 1.0 / (distance_matrix + 1e-9)
        
    # Use a symmetrized k-NN filtered inverse distance heuristic.
    # This creates a sparse geometric graph similar to Delaunay but adapted to local density.
    # k=6 is chosen to approximate the average degree of a Delaunay triangulation for N=50.
    k = min(6, n - 1)
    
    # Initialize heuristic with very low values (effectively 0 for attraction)
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Compute k-nearest neighbors for each node
    # distance_matrix[i, j] is the distance between i and j
    # We need to find the k smallest distances for each row, excluding the diagonal (distance 0)
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback to simple inverse distance if time is running out
            return 1.0 / (distance_matrix + 1e-9)
            
        # Get distances from i to all other nodes
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # Exclude self
        
        # Find indices of k nearest neighbors
        # np.argpartition is faster than full sort for finding k smallest
        if k < n - 1:
            knn_indices = np.argpartition(dists, k)[:k]
        else:
            knn_indices = np.argsort(dists)[:k]
            
        # Set high attractiveness for these edges
        for j in knn_indices:
            d = distance_matrix[i, j]
            if d > 0:
                heuristic[i, j] = 1.0 / d
            else:
                heuristic[i, j] = 1e9 # Should not happen for i!=j
                
        # Symmetrize: if j is in i's k-NN, make i attractive from j too?
        # The directive/knowledge implies a filter. 
        # A common approach is to set eta[i,j] high if j in NN(i) OR i in NN(j).
        # I will do a second pass to symmetrize the mask.
        
    # Symmetrize the heuristic mask
    # If heuristic[i,j] > 1e-8 (meaning j was in i's kNN), 
    # we want heuristic[j,i] to also be high if it wasn't already.
    # However, the distance is the same, so 1/d is the same.
    # We just need to ensure the mask is symmetric.
    
    mask = heuristic > 1e-8
    # Symmetrize the mask
    mask = mask | mask.T
    
    # Reconstruct heuristic based on the symmetric mask
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = 1e-9
            elif mask[i, j]:
                d = distance_matrix[i, j]
                heuristic[i, j] = 1.0 / (d + 1e-9)
            else:
                heuristic[i, j] = 1e-9
                
    return heuristic
# EVOLVE-BLOCK-END
