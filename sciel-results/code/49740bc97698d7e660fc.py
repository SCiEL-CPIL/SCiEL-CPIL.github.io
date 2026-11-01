def reduce_rank(U_mat, V_mat, W_mat, tol=1e-10):
    n_terms = U_mat.shape[1]
    # Iterate to remove dependent terms
    while n_terms > 0:
        # Build matrix of flattened tensors
        M = np.zeros((U_mat.shape[0] * V_mat.shape[0] * W_mat.shape[0], n_terms))
        for r in range(n_terms):
            M[:, r] = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r]).flatten()
        
        rank_M = np.linalg.matrix_rank(M, tol=tol)
        
        if rank_M < n_terms:
            # Find null space
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            # Null space is spanned by last (n_terms - rank_M) rows of Vt
            # We want to eliminate one term.
            # Pick the null vector that allows eliminating a term with largest coefficient magnitude?
            # Or just pick the last row of Vt.
            # To maximize stability, we can try all null vectors?
            # For now, just use the last row.
            c = Vt[-1, :]
            
            # Find index to eliminate
            k_idx = np.argmax(np.abs(c))
            if np.abs(c[k_idx]) > tol:
                # Eliminate k_idx
                # New coeffs: 1 - c_i / c_k
                new_U = []
                new_V = []
                new_W = []
                for r in range(n_terms):
                    if r == k_idx:
                        continue
                    coeff = 1 - c[r] / c[k_idx]
                    new_U.append(U_mat[:, r] * coeff)
                    new_V.append(V_mat[:, r] * coeff)
                    new_W.append(W_mat[:, r] * coeff)
                
                U_mat = np.array(new_U).T
                V_mat = np.array(new_V).T
                W_mat = np.array(new_W).T
                n_terms -= 1
            else:
                break
        else:
            break
            
    return U_mat, V_mat, W_mat
