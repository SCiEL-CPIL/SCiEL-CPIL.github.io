def residuals(params, rank, n, m, p):
    U_flat = params[:n*m*rank]
    V_flat = params[n*m*rank : n*m*rank + m*p*rank]
    W_flat = params[n*m*rank + m*p*rank : ]
    
    U_mat = U_flat.reshape(n*m, rank)
    V_mat = V_flat.reshape(m*p, rank)
    W_mat = W_flat.reshape(p*n, rank)
    
    res = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                res[idx] = s - 1.0
                idx += 1
    return res
