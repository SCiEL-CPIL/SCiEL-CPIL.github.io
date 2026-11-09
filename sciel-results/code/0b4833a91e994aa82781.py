"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix (larger is more attractive) 
    of the same shape as distance_matrix.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # 1. Base heuristic: inverse distance
    # Avoid division by zero for diagonal
    d = distance_matrix.copy()
    np.fill_diagonal(d, 1e9)
    h_base = 1.0 / (d + 1e-9)
    
    # 2. Global heuristic: Hungarian Assignment Boost
    # Set diagonal to inf to prevent self-loops in assignment
    cost_mat = distance_matrix.copy()
    np.fill_diagonal(cost_mat, np.inf)
    
    try:
        # Find optimal cycle cover
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)
        
        # Create mask for edges in the optimal assignment
        in_cover = np.zeros((n, n), dtype=bool)
        in_cover[row_ind, col_ind] = True
        
        # Boost edges in the optimal cycle cover
        # A factor of 2.0 doubles the attractiveness of these edges
        # This encourages the ACO to explore the "good" global structure
        boost_factor = 2.0
        h_global = h_base * (1.0 + boost_factor * in_cover)
        
        # Combine: We can just use h_global
        # Or blend: 0.5 * h_base + 0.5 * h_global? 
        # h_global already contains h_base.
        # Let's just use h_global.
        
        result = h_global
        
    except Exception:
        # Fallback to base heuristic if Hungarian fails
        result = h_base

    # Ensure positive values
    result = np.maximum(result, 1e-9)
    
    # Safety check: ensure no NaN/Inf
    if not np.all(np.isfinite(result)):
        result = h_base
        result = np.maximum(result, 1e-9)
        
    return result
# EVOLVE-BLOCK-END
