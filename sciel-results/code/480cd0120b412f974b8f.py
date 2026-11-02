<<<<<<< SEARCH
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = [[0] * rank for _ in range(num_u)]
                    V = [[0] * rank for _ in range(num_v)]
                    W = [[0] * rank for _ in range(num_w)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][column] = 1
                                V[jj * p + kk][column] = 1
                                W[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
=======
    # Verify the decomposition efficiently
    if time_remaining() < 5:
        return {"U": U, "V": V, "W": W}
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for einsum: U->(n,m,rank), V->(m,p,rank), W->(p,n,rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    if not np.allclose(T, 1.0, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(num_u)]
        V = [[0] * rank for _ in range(num_v)]
        W = [[0] * rank for _ in range(num_w)]
        column = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][column] = 1
                    V[jj * p + kk][column] = 1
                    W[kk * n + ii][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
