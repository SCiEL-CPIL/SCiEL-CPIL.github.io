def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def strassen_block(u00, u01, u10, u11, v00, v01, v10, v11, w00, w01, w10, w11):
        nonlocal col
        # M1 = (p00+p11)(q00+q11)
        U[u00][col] = 1; U[u11][col] = 1
        V[v00][col] = 1; V[v11][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1
        # M2 = (p10+p11)q00
        U[u10][col] = 1; U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        col += 1
        # M3 = p00(q11-q00)
        U[u00][col] = 1
        V[v11][col] = 1; V[v00][col] = -1
        W[w01][col] = 1
        col += 1
        # M4 = p11(q10-q00)
        U[u11][col] = 1
        V[v10][col] = 1; V[v00][col] = -1
        W[w00][col] = 1; W[w10][col] = 1
        col += 1
        # M5 = (p00+p01)q11
        U[u00][col] = 1; U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1; W[w01][col] = 1
        col += 1
        # M6 = (p10-p00)(q00+q01)
        U[u10][col] = 1; U[u00][col] = -1
        V[v00][col] = 1; V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        # M7 = (p01-p11)(q10+q11)
        U[u01][col] = 1; U[u11][col] = -1
        V[v10][col] = 1; V[v11][col] = 1
        W[w00][col] = 1
        col += 1
    
    # Block 1: A1*B1a
    # A1: u00=0,a00; u01=1,a01; u10=4,a10; u11=5,a11
    # B1a: v00=0,b00; v01=1,b01; v10=4,b10; v11=5,b11
    # C[0:2,0:2]: w00=0; w01=3; w10=1; w11=4
    strassen_block(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    
    # Block 2: A1*B1b
    # B1b: v00=2,b02; v01=3,b03; v10=6,b12; v11=7,b13
    # C[0:2,2:4]: w00=6; w01=9; w10=7; w11=10
    strassen_block(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    
    # Block 3: A2*B2a
    # A2: u00=2,a02; u01=3,a03; u10=6,a12; u11=7,a13
    # B2a: v00=8,b20; v01=9,b21; v10=12,b30; v11=13,b31
    # C[0:2,0:2]: w00=0; w01=3; w10=1; w11=4
    strassen_block(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    
    # Block 4: A2*B2b
    # B2b: v00=10,b22; v01=11,b23; v10=14,b32; v11=15,b33
    # C[0:2,2:4]: w00=6; w01=9; w10=7; w11=10
    strassen_block(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    
    # Part 2: A_bot * B (16 terms)
    # A_bot: a20,a21,a22,a23 → U indices 8,9,10,11
    for j in range(4):
        for k in range(4):
            U[8+j][col] = 1
            V[j*4+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
