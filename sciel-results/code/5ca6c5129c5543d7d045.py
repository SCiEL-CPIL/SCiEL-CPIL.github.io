<<<<<<< SEARCH
def _verify(U, V, W, n=N, m=M, p=P):
    """Verify trilinear decomposition condition."""
    rank = len(U[0]) if U else 0
    U_arr = np.array([[float(x) for x in row] for row in U])
    V_arr = np.array([[float(x) for x in row] for row in V])
    W_arr = np.array([[float(x) for x in row] for row in W])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
=======
def _verify(U, V, W, n=N, m=M, p=P):
    """Verify trilinear decomposition condition using vectorized NumPy."""
    if not U or not V or not W:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    r = U_arr.shape[1]
    if V_arr.shape[1] != r or W_arr.shape[1] != r:
        return False
        
    # Reshape to 3D tensors: U[i, j, c], V[j, k, c], W[k, i, c]
    U_3d = U_arr.reshape(n, m, r)
    V_3d = V_arr.reshape(m, p, r)
    W_3d = W_arr.reshape(p, n, r)
    
    # Compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Result shape: (n, m, p)
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
        
    # Check if all entries are close to 1
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
        
    return True
>>>>>>> REPLACE

