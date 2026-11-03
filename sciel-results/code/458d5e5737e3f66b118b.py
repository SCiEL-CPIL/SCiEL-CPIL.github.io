def verify_fast(U, V, W):
    # U: (n*m, r) -> (n, m, r)
    # V: (m*p, r) -> (m, p, r)
    # W: (p*n, r) -> (p, n, r)
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    # Sum over r
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Note: W index is k*n+i. So W_reshaped[k,i,r].
    result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.all(result == 1)
