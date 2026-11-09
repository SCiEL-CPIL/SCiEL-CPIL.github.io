import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    # Base heuristic: inverse distance
    # Add small epsilon to avoid division by zero
    # The prompt says non-positive become 1e-9, but 1/d is always positive for d>0.
    # For d=0, 1/d is inf. We handle this by adding epsilon.
    # The seed used 1e-9.
    eps = 1e-9
    base_heuristic = 1.0 / (distance_matrix + eps)
    
    # Compute MST
    # distance_matrix is dense. Convert to sparse for MST? 
    # Or use scipy.sparse.csgraph.minimum_spanning_tree which accepts dense?
    # Actually, minimum_spanning_tree takes a graph. 
    # We can pass the dense matrix directly? 
    # Documentation says: "graph : {sparse matrix, dense matrix}"
    # So we can pass distance_matrix.
    
    mst = minimum_spanning_tree(distance_matrix)
    
    # mst is a sparse matrix (csc or csr).
    # We need to identify the edges.
    # mst.indices, mst.indptr for CSR?
    # Let's convert to COO for easy edge extraction.
    mst_coo = mst.tocoo()
    
    # Boost edges in MST
    # Factor 2.0 seems reasonable. 
    # It makes MST edges twice as attractive as non-MST edges of the same length.
    boost_factor = 2.0
    
    # Create a copy to modify
    heuristic = base_heuristic.copy()
    
    # Iterate over MST edges
    # mst_coo.row, mst_coo.col
    rows = mst_coo.row
    cols = mst_coo.col
    
    # For each edge (i, j) in MST, boost heuristic[i, j] and heuristic[j, i]
    # Since TSP is symmetric, and ACO uses symmetric pheromone usually?
    # The prompt says "pheromone[i, j] for each step i to j ... in travel direction only".
    # But the heuristic is a matrix. Usually symmetric.
    # The distance matrix is symmetric.
    # So we should boost both [i, j] and [j, i].
    
    for i, j in zip(rows, cols):
        heuristic[i, j] *= boost_factor
        heuristic[j, i] *= boost_factor
        
    # Ensure no negative or zero values (though 1/d is always positive)
    # The prompt says: "Heuristic entries that are not positive become 1e-9"
    # This might be a check performed by the evaluator on the returned matrix.
    # Or it might mean I should do it.
    # "Heuristic entries that are not positive become 1e-9"
    # If I return 0 or negative, it becomes 1e-9.
    # My values are all positive.
    
    return heuristic
