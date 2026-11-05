import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication using:
    - 3x3 block structure (schoolbook, rank 27)
    - 2x2 Strassen within each block (rank 7)
    - Total rank = 27 * 7 = 189
    
    Indexing convention:
    - U[i*6 + j, r] for A[i,j]
    - V[j*6 + k, r] for B[j,k]
    - W[k*6 + i, r] for C[i,k] (trace convention: W[k*n+i, r])
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen 2x2 factors (rank 7)
    # U2[4, 7], V2[4, 7], W2[4, 7]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # P1 = (a+d)(e+h)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # P2 = (c+d)e
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # P3 = a(f-h)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # P4 = d(g-e)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # P5 = (a+b)h
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Schoolbook 3x3 factors (rank 27)
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j][col] = 1
                V3[j*3 + k][col] = 1
                W3[k*3 + i][col] = 1
                col += 1
    
    # Combine: total_rank = 27 * 7 = 189
    total_rank = 189
    
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Sparse representation for efficiency
    # Precompute non-zero entries
    U2_nz = {}
    for i2 in range(4):
        for j2 in range(4):
            for r2 in range(7):
                if U2[i2*2 + j2][r2] != 0:
                    key = (i2, j2)
                    if key not in U2_nz:
                        U2_nz[key] = {}
                    U2_nz[key][r2] = U2[i2*2 + j2][r2]
    
    V2_nz = {}
    for j2 in range(4):
        for k2 in range(4):
            for r2 in range(7):
                if V2[j2*2 + k2][r2] != 0:
                    key = (j2, k2)
                    if key not in V2_nz:
                        V2_nz[key] = {}
                    V2_nz[key][r2] = V2[j2*2 + k2][r2]
    
    W2_nz = {}
    for k2 in range(4):
        for i2 in range(4):
            for r2 in range(7):
                if W2[k2*2 + i2][r2] != 0:
                    key = (k2, i2)
                    if key not in W2_nz:
                        W2_nz[key] = {}
                    W2_nz[key][r2] = W2[k2*2 + i2][r2]
    
    # Fill U, V, W
    for r3 in range(27):
        for r2 in range(7):
            r = r3 * 7 + r2
            
            # U[u_idx, r] where u_idx = i*6 + j, i = i3*2 + i2, j = j3*2 + j2
            for i3 in range(3):
                for j3 in range(3):
                    val3 = U3[i3*3 + j3][r3]
                    if val3 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u_idx = (i3*2 + i2)*6 + (j3*2 + j2)
                            val2 = U2[i2*2 + j2][r2]
                            if val2 != 0:
                                U[u_idx][r] = val3 * val2
            
            # V[v_idx, r] where v_idx = j*6 + k, j = j3*2 + j2, k = k3*2 + k2
            for j3 in range(3):
                for k3 in range(3):
                    val3 = V3[j3*3 + k3][r3]
                    if val3 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v_idx = (j3*2 + j2)*6 + (k3*2 + k2)
                            val2 = V2[j2*2 + k2][r2]
                            if val2 != 0:
                                V[v_idx][r] = val3 * val2
            
            # W[w_idx, r] where w_idx = k*6 + i, k = k3*2 + k2, i = i3*2 + i2
            for k3 in range(3):
                for i3 in range(3):
                    val3 = W3[k3*3 + i3][r3]
                    if val3 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w_idx = (k3*2 + k2)*6 + (i3*2 + i2)
                            val2 = W2[k2*2 + i2][r2]
                            if val2 != 0:
                                W[w_idx][r] = val3 * val2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
