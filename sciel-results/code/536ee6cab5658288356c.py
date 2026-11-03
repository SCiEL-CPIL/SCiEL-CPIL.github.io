def _construct_2x3x3():
    n, m, p = 2, 3, 3
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Try ranks from 13 to 18
    for r in range(13, 19):
        U, V, W = _als_decompose(T, r, max_iter=500)
        if U is not None:
            # Convert to list of lists for the interface
            U_list = [[0]*r for _ in range(n*m)]
            V_list = [[0]*r for _ in range(m*p)]
            W_list = [[0]*r for _ in range(p*n)]
            for i in range(n):
                for j in range(m):
                    for rr in range(r):
                        U_list[i*m+j][rr] = U[i, j, rr]
            for j in range(m):
                for k in range(p):
                    for rr in range(r):
                        V_list[j*p+k][rr] = V[j, k, rr]
            for k in range(p):
                for i in range(n):
                    for rr in range(r):
                        W_list[k*n+i][rr] = W[k, i, rr]
            return U_list, V_list, W_list
    return None
