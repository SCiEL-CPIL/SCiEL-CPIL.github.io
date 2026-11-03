def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy einsum."""
    import numpy as np
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    # Check T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for consistent (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True
