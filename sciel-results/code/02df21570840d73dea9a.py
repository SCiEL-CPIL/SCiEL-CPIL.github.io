<<<<<<< SEARCH
    # A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5]  (2x1x1, schoolbook rank 2)
    add_schoolbook([0, 1], [2], [4])

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify correctness
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]

    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback: full schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    # A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5]  (2x1x1, schoolbook rank 2)
    add_schoolbook([0, 1], [2], [4])

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify correctness of base decomposition
    def verify(U, V, W, rank):
        tensor = np.zeros((n, m, p))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        return np.allclose(tensor, 1.0, atol=1e-9)

    if not verify(U, V, W, rank):
        # Fallback: full schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Structured rank reduction: try to find if any rank-1 term can be eliminated
    # by checking for linear dependencies in the factor space
    
    # Strategy: For each pair of factors (a, b), check if their tensor contribution
    # can be expressed as a linear combination of other factors.
    # More efficiently: try random linear combinations and see if we can maintain
    # the decomposition with fewer terms.
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to eliminate one term at a time by solving for it in terms of others
    # For a term c, we need: U_c ⊗ V_c ⊗ W_c = sum_{c' != c} alpha_{c'} U_{c'} ⊗ V_{c'} ⊗ W_{c'}
    # This is equivalent to: the vectorized tensor u_c v_c w_c^T must lie in the span
    # of {u_{c'} v_{c'} w_{c'}^T : c' != c}
    
    def try_eliminate_term(U, V, W, rank, target_col):
        """Try to eliminate term target_col by expressing it as combo of others."""
        if rank <= 1:
            return None
        
        # Build matrix of other terms (vectorized)
        other_cols = [c for c in range(rank) if c != target_col]
        n_terms = len(other_cols)
        
        # Each term contributes a vector of size n*m*p
        vec_size = n * m * p
        
        # Build the system: find alpha such that sum alpha_c * vec(U_c, V_c, W_c) = vec(U_t, V_t, W_t)
        # vec(U_c, V_c, W_c) is the Kronecker product U_c ⊗ V_c ⊗ W_c reshaped
        
        # Target vector
        U_t = U[:, target_col].flatten()
        V_t = V[:, target_col].flatten()
        W_t = W[:, target_col].flatten()
        target_vec = np.kron(np.kron(U_t, V_t), W_t)
        
        # Check if target is zero (already eliminable)
        if np.allclose(target_vec, 0, atol=1e-10):
            # Can remove this term
            new_U = np.delete(U, target_col, axis=1)
            new_V = np.delete(V, target_col, axis=1)
            new_W = np.delete(W, target_col, axis=1)
            return (new_U, new_V, new_W, rank - 1)
        
        # Build matrix of other terms
        A_mat = np.zeros((vec_size, n_terms))
        for idx, c in enumerate(other_cols):
            U_c = U[:, c].flatten()
            V_c = V[:, c].flatten()
            W_c = W[:, c].flatten()
            A_mat[:, idx] = np.kron(np.kron(U_c, V_c), W_c)
        
        # Solve least squares
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, target_vec, rcond=1e-10)
        except:
            return None
        
        # Check if solution is accurate
        if len(residuals) > 0 and residuals[0] < 1e-8:
            # Verify the solution
            reconstructed = A_mat @ alpha
            if np.allclose(reconstructed, target_vec, atol=1e-8):
                # We can express target as combo of others
                # But we need to maintain integer/half-integer entries
                # For now, just check if we can drop it (if alpha are simple)
                # This is complex; let's just check if the term is redundant
                pass
        
        return None
    
    # Try eliminating each term
    for target_col in range(rank):
        if time.time() > deadline - 10:
            break
        
        result = try_eliminate_term(U, V, W, rank, target_col)
        if result is not None:
            new_U, new_V, new_W, new_rank = result
            if verify(new_U, new_V, new_W, new_rank):
                best_rank = new_rank
                best_U, best_V, best_W = new_U, new_V, new_W
                rank = new_rank
                U, V, W = new_U, new_V, new_W
                break  # Restart with new rank
    
    # Try random linear combinations to find dependencies
    # Generate random vectors and check if they create zero-sum contributions
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Pick a random subset of 2-3 terms and check if their weighted sum is zero
        subset_size = np.random.choice([2, 3])
        subset = np.random.choice(rank, size=subset_size, replace=False)
        
        # For 2 terms: check if U_a ⊗ V_a ⊗ W_a = ± U_b ⊗ V_b ⊗ W_b
        if subset_size == 2:
            a, b = subset
            vec_a = np.kron(np.kron(U[:, a].flatten(), V[:, a].flatten()), W[:, a].flatten())
            vec_b = np.kron(np.kron(U[:, b].flatten(), V[:, b].flatten()), W[:, b].flatten())
            if np.allclose(vec_a, vec_b, atol=1e-10) or np.allclose(vec_a, -vec_b, atol=1e-10):
                # These terms cancel or duplicate
                # Remove one
                new_U = np.delete(U, b, axis=1)
                new_V = np.delete(V, b, axis=1)
                new_W = np.delete(W, b, axis=1)
                if verify(new_U, new_V, new_W, rank - 1):
                    best_rank = rank - 1
                    best_U, best_V, best_W = new_U, new_V, new_W
                    rank -= 1
                    U, V, W = new_U, new_V, new_W
                    break
    
    # Use the best result found
    if best_rank < 28:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    
    # Final verification
    if not verify(U, V, W, rank):
        # Fallback to original rank-28
        U = best_U.copy() if best_rank == 28 else None
        if U is None:
            # Reconstruct rank-28
            U = np.zeros((n * m, 28))
            V = np.zeros((m * p, 28))
            W = np.zeros((p * n, 28))
            col = 0
            add_strassen(0, 1, 0, 1, 0, 1)
            add_strassen(0, 1, 0, 1, 2, 3)
            add_schoolbook([0, 1], [0, 1], [4])
            add_schoolbook([0, 1], [2], [0, 1])
            add_schoolbook([0, 1], [2], [2, 3])
            add_schoolbook([0, 1], [2], [4])
            rank = 28
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
