def _block_decomposition_6x6x6_rank161():
    """
    Decompose 6x6x6 using 2x2 grid of 3x3 blocks with Strassen at block level.
    Block Strassen rank 7, each block mult is 3x3x3 rank 23.
    Total rank: 7 * 23 = 161.
    """
    n, m, p = 6, 6, 6
    rank = 7 * 23
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    r23U, r23V, r23W = _rank23_3x3()
    
    # Strassen coefficients for 2x2 block multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M4
    
    # Strassen U coefficients: U[i2][j2] for block (i2,j2)
    # i2,j2 in {0,1} for the 2x2 block grid
    sU = [[0]*7 for _ in range(4)]  # sU[i2*2+j2][c]
    sV = [[0]*7 for _ in range(4)]  # sV[j2*2+k2][c]
    sW = [[0]*7 for _ in range(4)]  # sW[k2*2+i2][c]
    
    # M1: (A11+A22)(B11+B22)
    sU[0][0] = 1  # A11
    sU[3][0] = 1  # A22
    sV[0][0] = 1  # B11
    sV[3][0] = 1  # B22
    sW[0][0] = 1  # C11
    sW[3][0] = 1  # C22
    
    # M2: (A21+A22)B11
    sU[2][1] = 1  # A21
    sU[3][1] = 1  # A22
    sV[0][1] = 1  # B11
    sW[2][1] = 1  # C21
    
    # M3: A11(B21-B11)
    sU[0][2] = 1  # A11
    sV[2][2] = 1  # B21
    sV[0][2] = -1 # B11
    sW[1][2] = 1  # C12
    sW[3][2] = 1  # C22
    
    # M4: A22(B12-B22)
    sU[3][3] = 1  # A22
    sV[1][3] = 1  # B12
    sV[3][3] = -1 # B22
    sW[0][3] = 1  # C11
    sW[1][3] = 1  # C12
    
    # M5: (A11+A12)B22
    sU[0][4] = 1  # A11
    sU[1][4] = 1  # A12
    sV[3][4] = 1  # B22
    sW[0][4] = -1 # C11
    sW[2][4] = 1  # C22
    
    # M6: (A21-A11)(B11+B12)
    sU[2][5] = 1  # A21
    sU[0][5] = -1 # A11
    sV[0][5] = 1  # B11
    sV[1][5] = 1  # B12
    sW[2][5] = 1  # C21
    
    # M7: (A12-A22)(B21+B22)
    sU[1][6] = 1  # A12
    sU[3][6] = -1 # A22
    sV[2][6] = 1  # B21
    sV[3][6] = 1  # B22
    sW[1][6] = 1  # C12
    
    # Now construct the full 6x6x6 decomposition
    col = 0
    for c in range(7):  # Strassen term
        for c23 in range(23):  # 3x3x3 term
            col_idx = c * 23 + c23
            
            # U: for each block (i2,j2) with sU[i2*2+j2][c] != 0
            for i2 in range(2):
                for j2 in range(2):
                    if sU[i2*2+j2][c] != 0:
                        for i3 in range(3):
                            for j3 in range(3):
                                u_row = (i2*3+i3)*6 + (j2*3+j3)
                                U[u_row][col_idx] = sU[i2*2+j2][c] * r23U[i3*3+j3][c23]
            
            # V: for each block (j2,k2) with sV[j2*2+k2][c] != 0
            for j2 in range(2):
                for k2 in range(2):
                    if sV[j2*2+k2][c] != 0:
                        for j3 in range(3):
                            for k3 in range(3):
                                v_row = (j2*3+j3)*6 + (k2*3+k3)
                                V[v_row][col_idx] = sV[j2*2+k2][c] * r23V[j3*3+k3][c23]
            
            # W: for each block (k2,i2) with sW[k2*2+i2][c] != 0
            for k2 in range(2):
                for i2 in range(2):
                    if sW[k2*2+i2][c] != 0:
                        for k3 in range(3):
                            for i3 in range(3):
                                w_row = (k2*3+k3)*6 + (i2*3+i3)
                                W[w_row][col_idx] = sW[k2*2+i2][c] * r23W[k3*3+i3][c23]
    
    assert col == 7 * 23
    return {"U": U, "V": V, "W": W}
