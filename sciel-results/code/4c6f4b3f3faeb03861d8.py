"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    # Strassen 2x2 decomposition
    # A indices: i*2+j, B indices: j*2+k, C indices: k*2+i
    # U: 4x7, V: 4x7, W: 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # P1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    # P2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    # P3 = A00(B10-B11)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    # P4 = A11(B01-B00)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    # P5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    # P6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    # P7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    
    # C00 = P1 + P4 - P5 + P7
    W2[0][0] = 1; W2[0][3] = 1; W2[0][4] = -1; W2[0][6] = 1
    # C01 = P3 + P5
    W2[1][2] = 1; W2[1][4] = 1
    # C10 = P2 + P4
    W2[2][1] = 1; W2[2][3] = 1
    # C11 = P1 - P2 + P3 + P6
    W2[3][0] = 1; W2[3][1] = -1; W2[3][2] = 1; W2[3][5] = 1
    
    # Tensor square to 4x4
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for a1 in range(7):
        for a0 in range(7):
            c = a1 * 7 + a0
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            val = U2[i0*2+j0][a0] * U2[i1*2+j1][a1]
                            if val != 0:
                                U[(i1*2+i0)*4 + (j1*2+j0)][c] = val
                            
                            val_v = V2[j0*2+0][a0] * V2[j1*2+0][a1]  # placeholder, will fix
                            
    # Redo more carefully
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for a1 in range(7):
        for a0 in range(7):
            c = a1 * 7 + a0
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            # U
                            val_u = U2[i0*2+j0][a0] * U2[i1*2+j1][a1]
                            if val_u != 0:
                                U[(i1*2+i0)*4 + (j1*2+j0)][c] = val_u
                            
                            # V: B index is (j,k), j = j1*2+j0, k = k1*2+k0
                            for k1 in range(2):
                                for k0 in range(2):
                                    val_v = V2[j0*2+k0][a0] * V2[j1*2+k1][a1]
                                    if val_v != 0:
                                        V[(j1*2+j0)*4 + (k1*2+k0)][c] = val_v
                            
                            # W: C index is (k,i), k = k1*2+k0, i = i1*2+i0
                            for k1 in range(2):
                                for k0 in range(2):
                                    val_w = W2[k0*2+i0][a0] * W2[k1*2+i1][a1]
                                    if val_w != 0:
                                        W[(k1*2+k0)*4 + (i1*2+i0)][c] = val_w
    
    # Verify correctness
    if time.time() - start_time < 140:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal 1 if i,j,k are valid (always for matrix mult), else 0
        # Actually for matrix multiplication, the tensor is T[i,j,k] = 1 for all i,j,k
        # since C[i,k] = sum_j A[i,j]*B[j,k]
        
        # Check: for each (i,j,k), sum_c U[i*4+j,c]*V[j*4+k,c]*W[k*4+i,c] should be 1
        correct = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for c in range(49):
                        s += U_np[i*4+j, c] * V_np[j*4+k, c] * W_np[k*4+i, c]
                    if abs(s - 1.0) > 1e-9:
                        correct = False
                        break
            if not correct:
                break
        
        if not correct:
            # Fallback to schoolbook
            rank = 64
            U = [[0]*rank for _ in range(16)]
            V = [[0]*rank for _ in range(16)]
            W = [[0]*rank for _ in range(16)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U[i*4+j][col] = 1
                        V[j*4+k][col] = 1
                        W[k*4+i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
