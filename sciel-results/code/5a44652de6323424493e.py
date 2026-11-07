import numpy as np
import scipy.sparse.csgraph as csgraph

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    
    # Compute MST using scipy
    # csgraph.minimum_spanning_tree expects a sparse matrix or distance matrix
    # It returns a sparse matrix with 1s on MST edges
    # We need to ensure the graph is complete and undirected
    # distance_matrix is already complete and symmetric
    
    # Convert to sparse for MST algorithm
    # Note: scipy's minimum_spanning_tree works on the graph defined by the distances
    # It finds the subset of edges with minimum total weight
    mst = csgraph.minimum_spanning_tree(distance_matrix)
    
    # mst is a sparse matrix. Convert to boolean array
    # mst[i, j] > 0 means edge (i, j) is in the MST
    # Since the input is a distance matrix, the MST output will have the distances as weights?
    # No, scipy.sparse.csgraph.minimum_spanning_tree returns a sparse matrix 
    # where the values are the weights of the edges in the MST.
    # So we just need to check where the values are non-zero.
    
    mst_mask = (mst.toarray() > 0)
    
    # Initialize heuristic matrix
    # We want high constant for MST edges, low for others.
    # Let's try High = 10.0, Low = 1.0.
    # Or maybe scale it relative to the average distance?
    # The directive suggests "high constant" and "low constant".
    
    H = np.ones((n, n))
    H[mst_mask] = 10.0
    
    # Ensure diagonal is 0 or small? 
    # The problem says "Heuristic entries that are not positive become 1e-9".
    # Diagonals are self-loops, distance 0. 
    # In TSP ACO, we don't pick self-loops because they are already visited or not allowed.
    # But to be safe, let's set diagonal to 0.0. 
    # Wait, if H[i,i] is 0, it becomes 1e-9. That's fine.
    np.fill_diagonal(H, 0.0)
    
    return H
