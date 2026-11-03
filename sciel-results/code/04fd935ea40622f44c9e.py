"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Tensor product decomposition: R(6) <= R(2) * R(3) = 4 * 27 = 108
    Using schoolbook decompositions for 2x2x2 and 3x3x3.
    
    Indexing convention:
    - U[i*m + j, c] for A[i,j]
    - V[j*p + k, c] for B[j,k]  
    - W[k*n + i, c] for C[k,i] (trace convention: W[k*n+i, r])
    
    For 6x6x6 with 2x3 factorization:
    - i = i2*3 + i3, j = j2*3 + j3, k = k2*3 + k3
    - c = c2*27 + c3 where c2 in [0,4), c3 in [0,27)
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook 2x2x2: rank 4
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = [[0]*4 for _ in range(4)]  # 4x4
    V2 = [[0]*4 for _ in range(4)]  # 4x4
    W2 = [[0]*4 for _ in range(4)]  # 4x4
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U2[i*2+j][c] = 1
                V2[j*2+k][c] = 1
                W2[k*2+i][c] = 1
                c += 1
    
    # Schoolbook 3x3x3: rank 27
    # U3[i*3+j, c], V3[j*3+k, c], W3[k*3+i, c]
    U3 = [[0]*27 for _ in range(9)]  # 9x27
    V3 = [[0]*27 for _ in range(9)]  # 9x27
    W3 = [[0]*27 for _ in range(9)]  # 9x27
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][c] = 1
                V3[j*3+k][c] = 1
                W3[k*3+i][c] = 1
                c += 1
    
    # Combine via tensor product: rank = 4 * 27 = 108
    rank = 4 * 27
    U6 = [[0]*rank for _ in range(n*m)]  # 36 x 108
    V6 = [[0]*rank for _ in range(m*p)]  # 36 x 108
    W6 = [[0]*rank for _ in range(p*n)]  # 36 x 108
    
    for i2 in range(2):
        for j2 in range(2):
            for k2 in range(2):
                for c2 in range(4):
                    u2_val = U2[i2*2+j2][c2]
                    v2_val = V2[j2*2+k2][c2]
                    w2_val = W2[k2*2+i2][c2]
                    if u2_val == 0 and v2_val == 0 and w2_val == 0:
                        continue
                    for i3 in range(3):
                        for j3 in range(3):
                            for k3 in range(3):
                                for c3 in range(27):
                                    u3_val = U3[i3*3+j3][c3]
                                    v3_val = V3[j3*3+k3][c3]
                                    w3_val = W3[k3*3+i3][c3]
                                    if u3_val == 0 and v3_val == 0 and w3_val == 0:
                                        continue
                                    
                                    u_combined = u2_val * u3_val
                                    v_combined = v2_val * v3_val
                                    w_combined = w2_val * w3_val
                                    
                                    if u_combined == 0 and v_combined == 0 and w_combined == 0:
                                        continue
                                    
                                    # Row/col indices for 6x6
                                    i = i2*3 + i3
                                    j = j2*3 + j3
                                    k = k2*3 + k3
                                    
                                    col = c2*27 + c3
                                    
                                    U6[i*6+j][col] += u_combined
                                    V6[j*6+k][col] += v_combined
                                    W6[k*6+i][col] += w_combined
    
    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
