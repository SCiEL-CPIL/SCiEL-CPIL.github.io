<<<<<<< SEARCH
    # Attempt rank reduction: try removing columns and verifying
    # This is a greedy approach that may find linear dependencies
    if time.time() < deadline - 30:
        removed = []
        for r in range(rank_bs):
            if time.time() > deadline - 10:
                break
            mask = np.ones(rank_bs, dtype=bool)
            mask[r] = False
            U_test = U_bs[:, mask]
            V_test = V_bs[:, mask]
            W_test = W_bs[:, mask]
            
            if verify(U_test, V_test, W_test):
                removed.append(r)
        
        if removed:
            mask = np.ones(rank_bs, dtype=bool)
            mask[removed] = False
            best_U = U_bs[:, mask]
            best_V = V_bs[:, mask]
            best_W = W_bs[:, mask]
            best_rank = rank_bs - len(removed)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
=======
    # Attempt rank reduction: find linear dependencies among rank-1 slices
    # The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # A set of columns is redundant iff the corresponding tensor is zero,
    # which is equivalent to the columns being linearly dependent in a
    # suitable "flattened" representation.
    #
    # Efficient check: for each candidate removal, verify via the tensor
    # contraction. But a faster necessary-and-sufficient test for a single
    # column removal: the tensor contribution of that column must be
    # expressible as a linear combination of the others.
    #
    # Practical approach: build the full tensor T (216 elements) from the
    # decomposition, then check if removing a column still reconstructs T.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    if time.time() < deadline - 30:
        # Build the full tensor T of shape (n, m, p)
        U_flat = U_bs  # (n*m, R)
        V_flat = V_bs  # (m*p, R)
        W_flat = W_bs  # (p*n, R)
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape: U -> (n, m, R), V -> (m, p, R), W -> (p, n, R)
        U_3d = U_flat.reshape(n, m, rank_bs)
        V_3d = V_flat.reshape(m, p, rank_bs)
        W_3d = W_flat.reshape(p, n, rank_bs)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T_full = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # For each column r, compute its contribution tensor:
        # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
        # If T_r is in the span of {T_{r'} for r' != r}, we can remove r.
        
        # Build a matrix whose columns are the flattened T_r vectors.
        # T_r has n*m*p = 216 elements.
        R = rank_bs
        tensor_dim = n * m * p
        T_matrix = np.zeros((tensor_dim, R), dtype=np.float64)
        for r in range(R):
            T_r = U_3d[:, :, r] * V_3d[:, :, r] * W_3d[:, :, r]  # (n,m,p)
            T_matrix[:, r] = T_r.reshape(tensor_dim)
        
        # Check linear dependencies: a column r is removable iff
        # T_matrix[:, r] is in the column space of T_matrix with column r removed.
        # Equivalent: rank(T_matrix) == rank(T_matrix without col r)
        
        # Compute rank of full matrix
        # Use SVD for numerical stability
        try:
            U_svd, s_vals, Vt = np.linalg.svd(T_matrix, full_matrices=False)
            full_rank = np.sum(s_vals > 1e-9)
        except np.linalg.LinAlgError:
            full_rank = R
        
        # For each column, check if removing it drops the rank
        removed = []
        for r in range(R):
            if time.time() > deadline - 10:
                break
            # Remove column r
            mask = np.ones(R, dtype=bool)
            mask[r] = False
            T_reduced = T_matrix[:, mask]
            
            try:
                U_r, s_r, Vt_r = np.linalg.svd(T_reduced, full_matrices=False)
                red_rank = np.sum(s_r > 1e-9)
            except np.linalg.LinAlgError:
                red_rank = R - 1
            
            if red_rank == full_rank:
                removed.append(r)
        
        if removed:
            mask = np.ones(R, dtype=bool)
            mask[removed] = False
            best_U = U_bs[:, mask]
            best_V = V_bs[:, mask]
            best_W = W_bs[:, mask]
            best_rank = R - len(removed)
            
            # Final verification
            if not verify(best_U, best_V, best_W):
                # Shouldn't happen, but fallback
                best_U, best_V, best_W = U_bs.copy(), V_bs.copy(), W_bs.copy()
                best_rank = R
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
>>>>>>> REPLACE
