def _verify(U, V, W):
    """Check exact decomposition on random integer matrices."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randint(-5, 6, (n, m)).astype(float)
        B = rng.randint(-5, 6, (m, p)).astype(float)
        C = A @ B
        
        # Efficient computation:
        # C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Let U_mat be (n*m, rank), V_mat be (m*p, rank), W_mat be (p*n, rank)
        # We can reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        # But W index is (k*n+i). So W is (p, n, rank) if we reshape (p*n, rank) to (p, n, rank).
        # Let's check indices:
        # U[idx, c] where idx = i*m + j. So U is (n, m, rank).
        # V[idx, c] where idx = j*p + k. So V is (m, p, rank).
        # W[idx, c] where idx = k*n + i. So W is (p, n, rank).
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # C[i,k] = sum_c sum_j U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # This is a tensor contraction.
        # We can do:
        # Step 1: Contract U and V over j and c? No, c is the rank index we sum over.
        # Let M1 = sum_j U_r[i,j,c] * V_r[j,k,c] -> (n, p, rank)
        # Then C[i,k] = sum_c M1[i,k,c] * W_r[k,i,c]
        
        # Compute M1:
        # U_r: (n, m, rank)
        # V_r: (m, p, rank)
        # We want M1[i, k, c] = sum_j U_r[i, j, c] * V_r[j, k, c]
        # This is like a batched matmul if we rearrange.
        # U_r.transpose(0, 2, 1) -> (n, rank, m)
        # V_r.transpose(2, 0, 1) -> (rank, m, p)
        # M1_batch = np.einsum('nrm,mpr->np r', U_r, V_r) ? No.
        # M1[i,k,c] = sum_j U[i,j,c] V[j,k,c]
        # Use einsum:
        M1 = np.einsum('ijc,jkc->ikc', U_r, V_r)
        
        # Now C[i,k] = sum_c M1[i,k,c] * W_r[k,i,c]
        # W_r is (p, n, rank). We need W_r[k,i,c].
        # So we need to transpose W_r to (n, p, rank) to match i,k,c order?
        # W_r[k, i, c]. We want to sum over c.
        # C[i,k] = sum_c M1[i,k,c] * W_r[k,i,c]
        # Let's use einsum:
        C_calc = np.einsum('ikc,kic->ik', M1, W_r)
        
        if not np.allclose(C, C_calc, atol=1e-6):
            return False
    return True
