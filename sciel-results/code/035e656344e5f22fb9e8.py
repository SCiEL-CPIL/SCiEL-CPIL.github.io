<<<<<<< SEARCH
def _try_linear_combo_remove(U, V, W, col_remove, n, m, p, deadline):
    """Try to express col_remove as linear combination of others."""
    current_rank = len(U[0])
    
    # Build the tensor contribution of col_remove
    U_rm = np.array([row[col_remove] for row in U], dtype=np.float64)
    V_rm = np.array([row[col_remove] for row in V], dtype=np.float64)
    W_rm = np.array([row[col_remove] for row in W], dtype=np.float64)
    
    # Try to find coefficients c_j such that sum_j c_j * (U_j * V_j * W_j) = U_rm * V_rm * W_rm
    # This is a nonlinear problem, so use heuristic search
    
    # Simple approach: try each other column as a candidate to absorb
    for col_keep in range(current_rank):
        if col_keep == col_remove:
            continue
        
        U_kp = np.array([row[col_keep] for row in U], dtype=np.float64)
        V_kp = np.array([row[col_keep] for row in V], dtype=np.float64)
        W_kp = np.array([row[col_keep] for row in W], dtype=np.float64)
        
        # Check if col_remove is proportional to col_keep in a useful way
        # Try various scales
        for scale in [-2, -1, -0.5, 0.5, 1, 2]:
            result = _try_scale_and_remove(U, V, W, col_keep, col_remove, scale, n, m, p)
            if result is not None:
                return result
    
    return None
=======
def _try_linear_combo_remove(U, V, W, col_remove, n, m, p, deadline):
    """Try to express col_remove as linear combination of others."""
    current_rank = len(U[0])
    
    # Try pairwise merges with other columns
    for col_keep in range(current_rank):
        if col_keep == col_remove:
            continue
        
        # Try merging col_remove into col_keep
        result = _try_pairwise_merge(U, V, W, col_remove, col_keep, n, m, p)
        if result is not None:
            return result
        
        # Try merging col_keep into col_remove
        result = _try_pairwise_merge(U, V, W, col_keep, col_remove, n, m, p)
        if result is not None:
            return result
        
        # Also try simple scaling approach
        for scale in [-2, -1, -0.5, 0.5, 1, 2]:
            result = _try_scale_and_remove(U, V, W, col_keep, col_remove, scale, n, m, p)
            if result is not None:
                return result
    
    return None
>>>>>>> REPLACE

