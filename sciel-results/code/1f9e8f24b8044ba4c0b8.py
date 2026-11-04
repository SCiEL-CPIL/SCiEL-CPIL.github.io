"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Use Kronecker product structure with Strassen for 2x2x2 blocks
    # 4x4x4 can be done with rank 49 (7x7 Kronecker of Strassen)
    # 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1, rank = 49 + 49 + 4 = 102
    
    # Strassen 2x2x2 decomposition
    # M1 = (A11+A22)(B12-B22)
    # M2 = (A11+A22)(B22)
    # M3 = (A11)(B11-B21)
    # M4 = (A22)(B12-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A22)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M1 + M2
    # C22 = M2 - M3 + M4 + M6
    
    # For 2x2x2, rank 7:
    # U rows indexed by (i,j) for i in {0,1}, j in {0,1}
    # V rows indexed by (j,k) for j in {0,1}, k in {0,1}
    # W rows indexed by (k,i) for k in {0,1}, i in {0,1}
    
    strassen_U = [
        [0, 0, 1, 0, 1, 0, 0],  # (0,0)
        [0, 0, 0, 0, 0, 1, 0],  # (0,1)
        [0, 0, 0, 0, 0, 0, 1],  # (1,0)
        [0, 0, 1, 0, 0, 0, 0],  # (1,1)
    ]
    strassen_V = [
        [1, 0, 0, 0, 0, 1, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 1],  # (0,1)
        [0, 0, 1, 1, 0, 0, 0],  # (1,0)
        [0, 0, 0, 1, 1, 0, 0],  # (1,1)
    ]
    strassen_W = [
        [0, 1, 0, 0, 0, 0, 1],  # (0,0)
        [0, 0, 1, 0, 0, 0, 0],  # (0,1)
        [1, 0, 0, 0, 0, 1, 0],  # (1,0)
        [0, 0, 0, 1, 0, 0, 0],  # (1,1)
    ]
    
    # Verify Strassen: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal 1 if i,j,k form a valid product term
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k in {0,1} (it's the multiplication tensor)
    # The decomposition should satisfy: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # Let me verify this is correct by checking
    sU = np.array(strassen_U)
    sV = np.array(strassen_V)
    sW = np.array(strassen_W)
    
    # Check: for each (i,j,k), sum over c of U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for c in range(7):
                    val += sU[i*2+j, c] * sV[j*2+k, c] * sW[k*2+i, c]
                if val != 1:
                    print(f"Strassen check failed for ({i},{j},{k}): got {val}")
    
    # Now build 4x4x4 using Kronecker of two Strassen decompositions
    # Rank = 7 * 7 = 49
    # U4x4x4[i*4+j, c1*7+c2] = sU[i2*2+j2, c2] * sU[i1*2+j1, c1] where i=i1*2+i2, j=j1*2+j2
    # V4x4x4[j*4+k, c1*7+c2] = sV[j2*2+k2, c2] * sV[j1*2+k1, c1] where j=j1*2+j2, k=k1*2+k2
    # W4x4x4[k*4+i, c1*7+c2] = sW[k2*2+i2, c2] * sW[k1*2+i1, c1] where k=k1*2+k2, i=i1*2+i2
    
    # Build 4x4x4 decomposition
    U444 = [[0]*49 for _ in range(16)]
    V444 = [[0]*49 for _ in range(16)]
    W444 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for c1 in range(7):
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                for c2 in range(7):
                                    c = c1 * 7 + c2
                                    i = i1 * 2 + i2
                                    j = j1 * 2 + j2
                                    k = k1 * 2 + k2
                                    uval = sU[i2*2+j2, c2] * sU[i1*2+j1, c1]
                                    vval = sV[j2*2+k2, c2] * sV[j1*2+k1, c1]
                                    wval = sW[k2*2+i2, c2] * sW[k1*2+i1, c1]
                                    U444[i*4+j][c] += uval
                                    V444[j*4+k][c] += vval
                                    W444[k*4+i][c] += wval
    
    # Verify 4x4x4
    U444_arr = np.array(U444, dtype=float)
    V444_arr = np.array(V444, dtype=float)
    W444_arr = np.array(W444, dtype=float)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for c in range(49):
                    val += U444_arr[i*4+j, c] * V444_arr[j*4+k, c] * W444_arr[k*4+i, c]
                if abs(val - 1) > 1e-9:
                    print(f"4x4x4 check failed for ({i},{j},{k}): got {val}")
    
    # Now for 4x4x9: use 4x4x4 for first 8 columns (k=0..7) and 4x4x1 for column k=8
    # 4x4x1: rank 4, U[i,j,c]=1, V[j,0,c]=1, W[0,i,c]=1 for c=i
    
    # Total rank = 49 + 49 + 4 = 102
    total_rank = 102
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]  # m*p = 4*9 = 36
    W = [[0]*total_rank for _ in range(36)]  # p*n = 9*4 = 36
    
    # First 49 columns: 4x4x4 for k=0..3 (k1=0)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    U[i*4+j][c] = U444[i*4+j][c]
                    V[j*4+k][c] = V444[j*4+k][c]
                    W[k*4+i][c] = W444[k*4+i][c]
    
    # Next 49 columns: 4x4x4 for k=4..7 (k1=1), offset by 4 in k
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    U[i*4+j][49+c] = U444[i*4+j][c]
                    V[j*4+(k+4)][49+c] = V444[j*4+k][c]
                    W[(k+4)*4+i][49+c] = W444[k*4+i][c]
    
    # Last 4 columns: 4x4x1 for k=8
    for i in range(4):
        for j in range(4):
            for c in range(4):
                U[i*4+j][98+c] = 1.0
                V[j*4+8][98+c] = 1.0 if j == c else 0.0
                W[8*4+i][98+c] = 1.0 if i == c else 0.0
    
    # Final verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(9):
                val = 0
                for c in range(total_rank):
                    val += U_arr[i*4+j, c] * V_arr[j*4+k, c] * W_arr[k*4+i, c]
                if k < 8:
                    expected = 1.0
                else:
                    expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
                if err > 1e-9:
                    print(f"Final check failed for ({i},{j},{k}): got {val}, expected {expected}")
    
    if max_err > 1e-9:
        print(f"Max error: {max_err}")
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
