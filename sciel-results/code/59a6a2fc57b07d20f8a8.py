def _find_M223_rank11(deadline):
    """Try to find a rank-11 decomposition of M_{2,2,3}."""
    n_block, m_block, p_block = 2, 2, 3
    T = np.ones((n_block, m_block, p_block))
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
        U = np.random.randn(n_block*m_block, 11)
        V = np.random.randn(m_block*p_block, 11)
        
        # Set up the linear system for W
        # For each (i,k): sum_r W[k*n_block+i, r] * (sum_j U[i*m_block+j, r] * V[j*p_block+k, r]) = 1
        B = np.zeros((n_block*p_block, 11))
        for i in range(n_block):
            for k in range(p_block):
                for r in range(11):
                    B[i*p_block+k, r] = sum(U[i*m_block+j, r] * V[j*p_block+k, r] for j in range(m_block))
        
        # The system is: for each (i,k), sum_r B[(i,k), r] * W[k*n_block+i, r] = 1
        # But W[k*n_block+i, r] is a single variable, so the system is:
        # B @ W_col = 1, where W_col is a vector of length n_block*p_block*11
        # Wait, this is not right. Let me redo.
        
        # The equation is: for each (i,k), sum_r B[(i,k), r] * W[k*n_block+i, r] = 1
        # This is n_block*p_block equations in n_block*p_block*11 unknowns
        # I can solve this using least squares
        
        # Reshape W to a vector
        W_vec = np.random.randn(n_block*p_block*11)
        # The system is: for each (i,k), sum_r B[(i,k), r] * W_vec[(k*n_block+i)*11 + r] = 1
        # This is a linear system
        
        # Let me set up the matrix A
        A = np.zeros((n_block*p_block, n_block*p_block*11))
        for i in range(n_block):
            for k in range(p_block):
                for r in range(11):
                    A[i*p_block+k, (k*n_block+i)*11 + r] = B[i*p_block+k, r]
        
        # Solve A @ W_vec = 1
        b = np.ones(n_block*p_block)
        try:
            W_vec_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            if residuals[0] < 1e-9:
                # Found a solution
                W = W_vec_sol.reshape(n_block*p_block, 11)
                # Reorder W to the correct shape
                W_reordered = np.zeros((p_block*n_block, 11))
                for k in range(p_block):
                    for i in range(n_block):
                        W_reordered[k*n_block+i, :] = W[i*p_block+k, :]
                return U.tolist(), V.tolist(), W_reordered.tolist()
        except:
            pass
    
    return None
