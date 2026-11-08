def _verify(U, V, W, n, m, p, tol=1e-9):
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # We need to check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Reshape U to (n, m, R)
    U_reshaped = U.reshape(n, m, -1)
    # Reshape V to (m, p, R)
    V_reshaped = V.reshape(m, p, -1)
    # Reshape W to (p, n, R)
    W_reshaped = W.reshape(p, n, -1)
    
    # We want C[i, k] = sum_j sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] ?
    # No, the tensor is T[i, j, k] = 1 if the term contributes to C[i, k] via A[i, j] and B[j, k].
    # The condition is: for all i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1.
    
    # Let's compute the tensor T_reconstructed[i, j, k]
    # U_idx = i*m+j
    # V_idx = j*p+k
    # W_idx = k*n+i
    
    # We can iterate over r (rank) and accumulate.
    # Or use einsum?
    # U is (n, m, R). V is (m, p, R). W is (p, n, R).
    # We want S[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r].
    
    # Note the indices:
    # U: i, j, r
    # V: j, k, r
    # W: k, i, r (Note: W is defined as W[k*n+i, r], so row index is k*n+i. 
    # If we reshape W to (p, n, R), then W_reshaped[k, i, r] corresponds to W[k*n+i, r].
    # So W_reshaped[k, i, r] is the correct element.
    
    # So S[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r].
    
    # We can compute this using einsum:
    # 'ijr,jkr,kir->ijk'
    # But wait, W is indexed by k, i.
    # Let's check the axes.
    # U: axis 0 is i, axis 1 is j, axis 2 is r.
    # V: axis 0 is j, axis 1 is k, axis 2 is r.
    # W: axis 0 is k, axis 1 is i, axis 2 is r.
    
    # We want to sum over r.
    # The result should be a tensor of shape (n, m, p) corresponding to (i, j, k).
    
    # Let's try:
    # np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Let's verify the indices for einsum.
    # U[i, j, r]
    # V[j, k, r]
    # W[k, i, r]
    # Sum over r.
    # Result index: i, j, k.
    
    # This matches!
    
    S = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if S is all 1s.
    if not np.allclose(S, 1.0, atol=tol):
        return False
    return True
