"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Kronecker product of 2x2 Strassen (rank 7) and 3x3 schoolbook (rank 27)
    # giving rank 7*27 = 108 for 6x6x6 matrix multiplication.
    
    # --- 2x2 Strassen decomposition (rank 7) ---
    # U2[a*2+b, c], V2[b*2+c, c], W2[c*2+a, c]
    # Strassen's 7 products:
    # M1 = (A11)(A22)(B11 - B22)
    # M2 = (A11 + A22)(B22)
    # M3 = (A22)(B11)
    # M4 = (A11 - A22)(B21 + B22)
    # M5 = (A11 + A12)(B11)
    # M6 = (A12 - A22)(B21 + B11)
    # M7 = (A21 + A11)(B12 - B22)
    # C11 = M1 + M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M1 + M2
    # C22 = M2 - M4 + M6 + M7
    
    U2 = [
        [1, 0, 0, 1, 1, 0, 1],   # A11 (a=0,b=0)
        [0, 0, 0, 0, 1, -1, 0],  # A12 (a=0,b=1)
        [0, 0, 1, 0, 0, 0, 1],   # A21 (a=1,b=0)
        [1, 1, 0, -1, 0, -1, 0], # A22 (a=1,b=1)
    ]
    V2 = [
        [1, 0, 0, 0, 0, 0, 0],   # B11 (b=0,c=0)
        [0, 0, 0, 0, 1, 0, 1],   # B12 (b=0,c=1)
        [-1, 1, 1, 1, 0, 1, -1], # B21 (b=1,c=0)
        [-1, 1, 0, 1, 0, 0, 0],  # B22 (b=1,c=1)
    ]
    W2 = [
        [1, 0, 0, 1, 1, 0, -1],  # C11 (c=0,a=0)
        [0, 0, 1, 0, 1, 0, 0],   # C12 (c=0,a=1)
        [1, 1, 0, 0, 0, 0, 0],   # C21 (c=1,a=0)
        [0, 1, 0, -1, 0, 1, 1],  # C22 (c=1,a=1)
    ]
    
    # --- 3x3 schoolbook decomposition (rank 27) ---
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
    
    # --- Kronecker product construction ---
    # 6x6 matrix indexed as: row = a*3+i, col = b*3+j where a,b in {0,1}, i,j in {0,1,2}
    # U[(a*3+i)*6 + (b*3+j), c*27+d] = U2[a*2+b, c] * U3[i*3+j, d]
    # V[(b*3+j)*6 + (c*3+k), c*27+d] = V2[b*2+c, c] * V3[j*3+k, d]
    # W[(c*3+k)*6 + (a*3+i), c*27+d] = W2[c*2+a, c] * W3[k*3+i, d]
    
    rank = 7 * 27  # 108
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for c in range(7):
        for d in range(27):
            idx = c * 27 + d
            for a in range(2):
                for b in range(2):
                    u2_val = U2[a*2+b][c]
                    v2_val = V2[b*2+c][c]
                    w2_val = W2[c*2+a][c]
                    if u2_val == 0 and v2_val == 0 and w2_val == 0:
                        continue
                    for i in range(3):
                        for j in range(3):
                            u3_val = U3[i*3+j][d]
                            if u2_val != 0 and u3_val != 0:
                                U[(a*3+i)*6 + (b*3+j)][idx] = u2_val * u3_val
                            for k in range(3):
                                v3_val = V3[j*3+k][d]
                                w3_val = W3[k*3+i][d]
                                if v2_val != 0 and v3_val != 0:
                                    V[(b*3+j)*6 + (c*3+k)][idx] = v2_val * v3_val
                                if w2_val != 0 and w3_val != 0:
                                    W[(c*3+k)*6 + (a*3+i)][idx] = w2_val * w3_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
