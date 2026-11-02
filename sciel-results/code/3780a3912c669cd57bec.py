def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(base_u, base_v, base_w, start_col):
        # base_u: [u00, u10, u01, u11] corresponding to A indices
        # base_v: [v00, v01, v10, v11] corresponding to B indices
        # base_w: [w00, w10, w01, w11] corresponding to C indices (k*n+i)
        
        # M1
        U[base_u[0]][start_col] += 1; V[base_v[0]][start_col] += 1; W[base_w[0]][start_col] += 1
        U[base_u[0]][start_col] += 1; V[base_v[1]][start_col] += 1; W[base_w[1]][start_col] += 1
        U[base_u[1]][start_col] += 1; V[base_v[0]][start_col] += 1; W[base_w[2]][start_col] -= 1
        U[base_u[1]][start_col] += 1; V[base_v[1]][start_col] += 1; W[base_w[3]][start_col] += 1
        
        # M2
        U[base_u[1]][start_col+1] += 1; V[base_v[0]][start_col+1] += 1; W[base_w[2]][start_col+1] += 1
        U[base_u[2]][start_col+1] += 1; V[base_v[0]][start_col+1] += 1; W[base_w[3]][start_col+1] -= 1
        
        # M3
        U[base_u[0]][start_col+2] += 1; V[base_v[1]][start_col+2] += 1; W[base_w[1]][start_col+2] += 1
        U[base_u[0]][start_col+2] += 1; V[base_v[3]][start_col+2] -= 1; W[base_w[3]][start_col+2] += 1
        
        # M4
        U[base_u[1]][start_col+3] += 1; V[base_v[2]][start_col+3] += 1; W[base_w[2]][start_col+3] += 1
        U[base_u[1]][start_col+3] += 1; V[base_v[0]][start_col+3] -= 1; W[base_w[0]][start_col+3] += 1
        
        # M5
        U[base_u[0]][start_col+4] += 1; V[base_v[3]][start_col+4] += 1; W[base_w[1]][start_col+4] -= 1
        U[base_u[1]][start_col+4] += 1; V[base_v[3]][start_col+4] += 1; W[base_w[3]][start_col+4] += 1
        
        # M6
        U[base_u[1]][start_col+5] += 1; V[base_v[0]][start_col+5] += 1; W[base_w[2]][start_col+5] += 1
        U[base_u[0]][start_col+5] -= 1; V[base_v[0]][start_col+5] += 1; W[base_w[0]][start_col+5] -= 1
        U[base_u[1]][start_col+5] += 1; V[base_v[1]][start_col+5] += 1; W[base_w[3]][start_col+5] += 1
        U[base_u[0]][start_col+5] -= 1; V[base_v[1]][start_col+5] += 1; W[base_w[1]][start_col+5] -= 1
        
        # M7
        U[base_u[1]][start_col+6] += 1; V[base_v[2]][start_col+6] += 1; W[base_w[1]][start_col+6] += 1
        U[base_u[2]][start_col+6] -= 1; V[base_v[2]][start_col+6] += 1; W[base_w[2]][start_col+6] -= 1
        U[base_u[1]][start_col+6] += 1; V[base_v[3]][start_col+6] += 1; W[base_w[3]][start_col+6] += 1
        U[base_u[2]][start_col+6] -= 1; V[base_v[3]][start_col+6] += 1; W[base_w[3]][start_col+6] -= 1

    def add_schoolbook_221(base_u, base_v, base_w, start_col):
        # A (2x2) * B (2x1) = C (2x1)
        # base_u: [u00, u10, u01, u11]
        # base_v: [v00, v10] (B is 2x1)
        # base_w: [w00, w10] (C is 2x1)
        
        # ae
        U[base_u[0]][start_col] += 1; V[base_v[0]][start_col] += 1; W[base_w[0]][start_col] += 1
        # bf
        U[base_u[1]][start_col+1] += 1; V[base_v[1]][start_col+1] += 1; W[base_w[0]][start_col+1] += 1
        # ce
        U[base_u[2]][start_col+2] += 1; V[base_v[0]][start_col+2] += 1; W[base_w[1]][start_col+2] += 1
        # df
        U[base_u[3]][start_col+3] += 1; V[base_v[1]][start_col+3] += 1; W[base_w[1]][start_col+3] += 1

    # Block 1: A1 (j=0,1), B1 (j=0,1)
    # A indices: 0,1 (i=0,j=0,1), 2,3 (i=1,j=0,1)
    # U indices: i*m+j. m=4.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 4. i=1, j=1 -> 5.
    u1 = [0, 4, 1, 5] # u00, u10, u01, u11
    
    # B indices: j*p+k. p=5.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # j=1, k=0 -> 5. j=1, k=1 -> 6.
    # j=0, k=2 -> 2. j=0, k=3 -> 3.
    # j=1, k=2 -> 7. j=1, k=3 -> 8.
    # j=0, k=4 -> 4. j=1, k=4 -> 9.
    
    # C indices: k*n+i. n=2.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    # k=2, i=0 -> 4. k=2, i=1 -> 5.
    # k=3, i=0 -> 6. k=3, i=1 -> 7.
    # k=4, i=0 -> 8. k=4, i=1 -> 9.
    
    # Strassen 1 (k=0,1)
    v1_s1 = [0, 2, 5, 7] # v00, v01, v10, v11
    w1_s1 = [0, 2, 1, 3] # w00, w10, w01, w11
    add_strassen(u1, v1_s1, w1_s1, 0)
    
    # Strassen 2 (k=2,3)
    v1_s2 = [2, 3, 7, 8]
    w1_s2 = [4, 6, 5, 7]
    add_strassen(u1, v1_s2, w1_s2, 7)
    
    # Schoolbook 1 (k=4)
    v1_s3 = [4, 9]
    w1_s3 = [8, 9]
    add_schoolbook_221(u1, v1_s3, w1_s3, 14)
    
    # Block 2: A2 (j=2,3), B2 (j=2,3)
    # U indices:
    # i=0, j=2 -> 2. i=0, j=3 -> 3.
    # i=1, j=2 -> 6. i=1, j=3 -> 7.
    u2 = [2, 6, 3, 7]
    
    # B indices:
    # j=2, k=0 -> 10. j=2, k=1 -> 11.
    # j=3, k=0 -> 15. j=3, k=1 -> 16.
    # j=2, k=2 -> 12. j=2, k=3 -> 13.
    # j=3, k=2 -> 17. j=3, k=3 -> 18.
    # j=2, k=4 -> 14. j=3, k=4 -> 19.
    
    # Strassen 3 (k=0,1)
    v2_s1 = [10, 12, 15, 17]
    w2_s1 = [0, 2, 1, 3]
    add_strassen(u2, v2_s1, w2_s1, 18)
    
    # Strassen 4 (k=2,3)
    v2_s2 = [12, 13, 17, 18]
    w2_s2 = [4, 6, 5, 7]
    add_strassen(u2, v2_s2, w2_s2, 25)
    
    # Schoolbook 2 (k=4)
    v2_s3 = [14, 19]
    w2_s3 = [8, 9]
    add_schoolbook_221(u2, v2_s3, w2_s3, 32)
    
    return {"U": U, "V": V, "W": W}
