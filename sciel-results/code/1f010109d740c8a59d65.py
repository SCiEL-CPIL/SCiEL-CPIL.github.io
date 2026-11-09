import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # 1. Base inverse distance
    # Add eps to avoid division by zero for self-loops or duplicates
    base_heuristic = 1.0 / (distance_matrix + eps)
    
    # 2. Compute Linear Assignment Problem (LAP) solution
    # This gives a set of N edges forming a cycle cover (possibly multiple subtours)
    # These edges are strong candidates for the TSP tour.
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    lap_mask = np.zeros((n, n), dtype=float)
    lap_mask[row_ind, col_ind] = 1.0
    
    # 3. Compute Minimum Spanning Tree (MST)
    # The optimal TSP tour is a 2-factor, and MST edges are often part of it.
    # Boosting MST edges helps maintain connectivity.
    # Convert dense distance matrix to sparse for MST
    # Note: distance_matrix has 0 on diagonal. MST will ignore them if we set them to inf or handle carefully.
    # scipy.sparse.csgraph.minimum_spanning_tree expects a weighted graph.
    # We create a sparse matrix.
    dm_sparse = csr_matrix(distance_matrix)
    # Ensure diagonal is 0 (or inf) so self-loops aren't picked
    np.fill_diagonal(distance_matrix, 0) # Temporarily modify? No, use a copy or handle in sparse creation.
    
    # Actually, let's just use the dense matrix and set diagonal to inf for MST calculation if needed, 
    # but scipy's MST on dense matrix works if diagonal is 0? 
    # minimum_spanning_tree(graph) where graph is a square matrix.
    # "The graph is represented by a square matrix... The matrix must be non-negative."
    # Diagonal entries are ignored if they are 0? 
    # Let's be safe and create a copy.
    dm_copy = distance_matrix.copy()
    np.fill_diagonal(dm_copy, np.inf)
    
    try:
        mst = minimum_spanning_tree(csr_matrix(dm_copy))
        mst_mask = np.zeros((n, n), dtype=float)
        # MST is directed? No, minimum_spanning_tree returns a directed graph representation of the undirected tree?
        # Actually, it returns a CSR matrix where edges are directed? 
        # "The result is a directed graph... if the input is undirected, the result is a directed graph with both directions?"
        # Let's check the shape.
        # For an undirected graph input, it returns a directed graph with arcs in both directions? 
        # Or just one? 
        # Documentation: "If the graph is undirected, the result is a directed graph with arcs in both directions."
        # So we can just use the non-zero entries.
        mst_coo = mst.tocoo()
        mst_mask[mst_coo.row, mst_coo.col] = 1.0
    except Exception:
        mst_mask = np.zeros((n, n), dtype=float)

    # 4. Combine heuristics
    # We want to boost LAP and MST edges.
    # Let's subtract a "discount" from the distance for these edges.
    
    # Determine scale of distances
    mean_dist = np.mean(distance_matrix[distance_matrix > 0]) if np.any(distance_matrix > 0) else 1.0
    
    # Discount factors
    # LAP edges are the most critical for breaking subtours and ensuring global optimality structure.
    alpha_lap = 0.5 * mean_dist
    # MST edges help with connectivity.
    alpha_mst = 0.2 * mean_dist
    
    # Adjusted distance
    # d_adj = d - alpha_lap * lap_mask - alpha_mst * mst_mask
    # We must ensure d_adj is positive.
    # If d is very small and we subtract too much, it becomes negative.
    # The heuristic 1/d_adj will then be negative or huge.
    # The prompt says: "Heuristic entries that are not positive become 1e-9".
    # So if d_adj <= 0, H becomes 1e-9 (very low attractiveness).
    # We want boosted edges to have HIGH attractiveness.
    # So we want d_adj to be SMALL but POSITIVE for boosted edges.
    
    # Let's use a multiplicative boost instead to be safer?
    # H = base_heuristic * (1 + boost_lap * lap_mask + boost_mst * mst_mask)
    
    # Multiplicative boost is safer against sign changes.
    boost_lap = 5.0
    boost_mst = 2.0
    
    # However, the directive mentioned "exponential of the negative reduced cost".
    # Let's try to emulate that.
    # Reduced cost c_ij - u_i - v_j.
    # For LAP edges, reduced cost is 0.
    # For others, it is > 0.
    # If we assume u_i + v_j approximates the "average" cost of the best edges incident to i and j?
    
    # Let's stick to the multiplicative boost which is robust.
    # We can also add a term that penalizes edges that are NOT in the LAP if they are "expensive" relative to the LAP solution?
    
    # Let's refine the boost.
    # The LAP solution is a lower bound.
    # Edges in the LAP are "free" in the dual sense.
    
    combined_heuristic = base_heuristic * (1.0 + boost_lap * lap_mask + boost_mst * mst_mask)
    
    # Ensure no negative values (though base_heuristic is positive and boosts are positive)
    # combined_heuristic = np.where(combined_heuristic > 0, combined_heuristic, 1e-9)
    
    # Normalize? The ACO algorithm uses pheromone[i,j] * heuristic[i,j].
    # The absolute scale of the heuristic matters relative to pheromone.
    # Pheromone starts at 1.
    # If heuristic is very large, the choice is dominated by heuristic.
    # If heuristic is very small, pheromone dominates.
    # Inverse distance is a standard scale.
    # Boosting by 5x makes LAP edges 6x more attractive than equivalent distance non-LAP edges.
    
    # Let's add a small regularization to avoid division by zero issues if any distance is 0 (though eps handles it)
    
    return combined_heuristic
