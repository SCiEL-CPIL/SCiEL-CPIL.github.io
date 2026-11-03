    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Vectorized computation of the tensor
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute T_computed
        # T[i, j, k] = sum_c U_reshaped[i, j, c] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
        # This is a contraction over c.
        # We can use einsum
        # T[i, j, k] = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        T_computed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        T_expected = np.ones((n, m, p))
        
        return np.max(np.abs(T_computed - T_expected)) < tol
