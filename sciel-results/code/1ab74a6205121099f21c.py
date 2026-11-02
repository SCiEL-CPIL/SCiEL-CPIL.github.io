def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify decomposition exactly using vectorized operations.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    R = U.shape[1]
    U_r = U.reshape(n, m, R)
    V_r = V.reshape(m, p, R)
    W_r = W.reshape(p, n, R)
    
    # Prod[i, j, k, r] = U[i, j, r] * V[j, k, r]
    # U_r: (n, m, R) -> (n, m, 1, R)
    # V_r: (m, p, R) -> (1, m, p, R)
    Prod = U_r[:, :, np.newaxis, :] * V_r[np.newaxis, :, :, :] # (n, m, p, R)
    
    # We need to multiply by W[k, i, r]
    # W_r is (p, n, R). We need to align k with axis 2 of Prod and i with axis 0 of Prod.
    # Transpose W_r to (n, p, R) so that index 0 is i and index 1 is k.
    W_t = W_r.T # (n, p, R)
    # Expand W_t to (n, 1, p, R) to broadcast with Prod (n, m, p, R)
    W_exp = W_t[:, np.newaxis, :, :]
    
    # Sum over r (axis 3)
    T_rec = np.sum(Prod * W_exp, axis=3) # (n, m, p)
    
    return np.allclose(T_rec, 1.0, atol=tol)
