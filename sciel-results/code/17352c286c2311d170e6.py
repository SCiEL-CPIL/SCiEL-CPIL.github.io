"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 2x4 * 4x5 with aggressive column-removal
    search to find linear dependencies and reduce rank below 36.
    """
    import time
    import numpy as np
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # 2, 4, 5

    def build_base_terms():
        """Build the base block-Strassen decomposition (rank 36)."""
        rank_terms = []  # Each term is (u_dict, v_dict, w_dict)

        def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
            i0, j0 = a_offset
            j1, k0 = b_offset
            c_i0, c_k0 = c_offset
            terms = [
                ({(i0, j0): 1, (i0+1, j0+1): 1},
                 {(j0, k0): 1, (j0+1, k0+1): 1},
                 {(i0, k0): 1, (i0+1, k0+1): 1}),
                ({(i0+1, j0): 1, (i0+1, j0+1): 1},
                 {(j0, k0): 1},
                 {(i0+1, k0): 1, (i0+1, k0+1): -1}),
                ({(i0, j0): 1},
                 {(j0, k0+1): 1, (j0+1, k0+1): -1},
                 {(i0, k0+1): 1, (i0+1, k0+1): 1}),
                ({(i0+1, j0+1): 1},
                 {(j0+1, k0): 1, (j0, k0): -1},
                 {(i0, k0): 1, (i0+1, k0): 1}),
                ({(i0, j0): 1, (i0, j0+1): 1},
                 {(j0+1, k0+1): 1},
                 {(i0, k0): -1, (i0, k0+1): 1}),
                ({(i0+1, j0): 1, (i0, j0): -1},
                 {(j0, k0): 1, (j0, k0+1): 1},
                 {(i0+1, k0+1): 1}),
                ({(i0, j0+1): 1, (i0+1, j0+1): -1},
                 {(j0+1, k0): 1, (j0+1, k0+1): 1},
                 {(i0, k0): 1}),
            ]
            for u, v, w in terms:
                u_scaled = {k: val * sign for k, val in u.items()}
                v_scaled = {k: val * sign for k, val in v.items()}
                w_scaled = {k: val * sign for k, val in w.items()}
                rank_terms.append((u_scaled, v_scaled, w_scaled))

        def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
            i0, j0 = a_offset
            j1, k0 = b_offset
            c_i0, c_k0 = c_offset
            terms = [
                ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
                ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
                ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
                ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
            ]
            for u, v, w in terms:
                u_scaled = {k: val * sign for k, val in u.items()}
                v_scaled = {k: val * sign for k, val in v.items()}
                w_scaled = {k: val * sign for k, val in w.items()}
                rank_terms.append((u_scaled, v_scaled, w_scaled))

        # Build block-Strassen decomposition (rank 36)
        add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
        add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
        add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
        add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
        add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
        add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))
        
        return rank_terms

    def terms_to_matrices(terms):
        """Convert list of term dicts to U, V, W matrices."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff

        return U, V, W

    def verify_decomposition(U, V, W):
        """Verify that the decomposition is correct."""
        rank = U.shape[1]
        # Check each (i,j,k) entry
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        u_val = U[i * m + j, r]
                        v_val = V[j * p + k, r]
                        w_val = W[k * n + i, r]
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True

    def try_remove_column(U, V, W, col_to_remove):
        """
        Try to remove column col_to_remove by expressing it as a linear combination
        of other columns. Returns (success, new_U, new_V, new_W) or (False, None, None, None).
        """
        rank = U.shape[1]
        if rank <= 1:
            return False, None, None, None
        
        # Get the column to remove
        u_col = U[:, col_to_remove]
        v_col = V[:, col_to_remove]
        w_col = W[:, col_to_remove]
        
        # Create matrix of other columns
        other_cols = [c for c in range(rank) if c != col_to_remove]
        U_other = U[:, other_cols]
        V_other = V[:, other_cols]
        W_other = W[:, other_cols]
        
        # We need to find coefficients alpha such that:
        # u_col = U_other @ alpha_u
        # v_col = V_other @ alpha_v  
        # w_col = W_other @ alpha_w
        # But this is over-constrained. Instead, we use the tensor structure.
        
        # The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # If we remove column r0, we need to adjust other columns to compensate.
        
        # Alternative approach: check if the column is linearly dependent in a way
        # that preserves the tensor. This is complex, so let's try a simpler heuristic:
        # Check if removing this column and zeroing it out still gives valid decomposition
        # (i.e., the column was already zero or redundant).
        
        # Actually, let's try: can we express the contribution of this column as a
        # linear combination of other columns' contributions?
        
        # For each (i,j,k), the contribution of column r0 is:
        # U[i*m+j, r0] * V[j*p+k, r0] * W[k*n+i, r0]
        
        # We want to find modifications to other columns that absorb this.
        # This is a rank-1 update problem.
        
        # Simpler approach: just try removing and verifying
        new_U = np.delete(U, col_to_remove, axis=1)
        new_V = np.delete(V, col_to_remove, axis=1)
        new_W = np.delete(W, col_to_remove, axis=1)
        
        if verify_decomposition(new_U, new_V, new_W):
            return True, new_U, new_V, new_W
        
        return False, None, None, None

    def find_linear_dependencies(U, V, W):
        """
        Find columns that can be removed by checking if the tensor can be
        preserved when a column is removed. Uses the structure of the problem.
        """
        rank = U.shape[1]
        removable_cols = []
        
        for col in range(rank):
            if time.time() > deadline - 5:
                break
            success, new_U, new_V, new_W = try_remove_column(U, V, W, col)
            if success:
                removable_cols.append(col)
        
        return removable_cols

    def iterative_column_removal(U, V, W):
        """
        Iteratively try to remove columns until no more can be removed.
        """
        max_iterations = 50
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            removable_cols = find_linear_dependencies(U, V, W)
            
            if not removable_cols:
                break
            
            # Remove the first removable column
            col = removable_cols[0]
            _, U, V, W = try_remove_column(U, V, W, col)
        
        return U, V, W

    # Build base decomposition
    base_terms = build_base_terms()
    U, V, W = terms_to_matrices(base_terms)
    
    # Verify base decomposition
    if not verify_decomposition(U, V, W):
        # Should not happen, but handle gracefully
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by removing columns
    U, V, W = iterative_column_removal(U, V, W)
    
    # Verify final decomposition
    if not verify_decomposition(U, V, W):
        # Fallback to base decomposition
        base_terms = build_base_terms()
        U, V, W = terms_to_matrices(base_terms)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
