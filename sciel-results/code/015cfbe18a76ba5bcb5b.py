<<<<<<< SEARCH
    # Part 2: Handle A[0:2, 2] * B[2:3, :]
    # This is 2x1 * 1x8 = 16 terms
    for i in range(2):
        for k in range(8):
            u_row = i*m + 2
            v_row = 2*p + k
            w_row = k*n + i
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1
            if time.time() > limit:
                rb = n*m*p
                U_f = [[0]*rb for _ in range(n*m)]
                V_f = [[0]*rb for _ in range(m*p)]
                W_f = [[0]*rb for _ in range(p*n)]
                c = 0
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            U_f[i*m+a][c] = 1
                            V_f[a*p+k][c] = 1
                            W_f[k*n+i][c] = 1
                            c += 1
                return {"U": U_f, "V": V_f, "W": W_f}
=======
    # Part 2: Handle A[0:2, 2] * B[2:3, :]
    # The tensor T[i,2,k] = 1 for all i in {0,1}, k in {0..7}
    # This is a rank-1 term: U[2][col]=1, U[5][col]=1, V[16+k][col]=1 for all k, W[2k+i][col]=1 for all i,k
    # Since W can depend on (i,k), we can use a single rank term
    U[0*m + 2][col] = 1
    U[1*m + 2][col] = 1
    for k in range(p):
        V[2*p + k][col] = 1
        for i in range(n):
            W[k*n + i][col] = 1
    col += 1
    if time.time() > limit:
        rb = n*m*p
        U_f = [[0]*rb for _ in range(n*m)]
        V_f = [[0]*rb for _ in range(m*p)]
        W_f = [[0]*rb for _ in range(p*n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U_f[i*m+a][c] = 1
                    V_f[a*p+k][c] = 1
                    W_f[k*n+i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
>>>>>>> REPLACE
