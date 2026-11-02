import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W

def _strassen_4x4():
    """
    Construct rank-49 decomposition for 4x4 * 4x4 using Kronecker product
    of two 2x2 * 2x2 Strassen decompositions.
    T_{4,4,4} = T_{2,2,2} (outer) x T_{2,2,2} (inner)
    """
    U1, V1, W1 = _strassen_2x2()
    U2, V2, W2 = _strassen_2x2()
    
    n1, m1, p1, r1 = 2, 2, 2, 7
    n2, m2, p2, r2 = 2, 2, 2, 7
    n, m, p = 4, 4, 4
    rank = r1 * r2 # 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            # U[i, j, c] = U1[i1, j1, c1] * U2[i2, j2, c2]
            # i = n1*i1 + i2, j = m1*j1 + j2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1, c1]
                    if u1_val == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2, c2]
                            if u2_val == 0: continue
                            i = n1 * i1 + i2
                            j = m1 * j1 + j2
                            U[i * m + j, c] = u1_val * u2_val

            # V[j, k, c] = V1[j1, k1, c1] * V2[j2, k2, c2]
            # j = m1*j1 + j2, k = p1*k1 + k2
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1, c1]
                    if v1_val == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2, c2]
                            if v2_val == 0: continue
                            j = m1 * j1 + j2
                            k = p1 * k1 + k2
                            V[j * p + k, c] = v1_val * v2_val

            # W[k, i, c] = W1[k1, i1, c1] * W2[k2, i2, c2]
            # k = p1*k1 + k2, i = n1*i1 + i2
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1, c1]
                    if w1_val == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2, c2]
                            if w2_val == 0: continue
                            k = p1 * k1 + k2
                            i = n1 * i1 + i2
                            W[k * n + i, c] = w1_val * w2_val

    return U, V, W

def construct():
    t_start = time.time()
    DEADLINE = t_start + 130
    
    n, m, p = DIMENSIONS # 4, 4, 8
    
    try:
        # Get 4x4x4 decomposition (rank 49)
        U4, V4, W4 = _strassen_4x4()
        
        # Construct 4x4x8 decomposition (rank 98)
        # Split B (4x8) into B_left (4x4) and B_right (4x4)
        # C (4x8) into C_left (4x4) and C_right (4x4)
        
        rank_half = 49
        rank_total = 98
        
        U = np.zeros((n * m, rank_total), dtype=float) # 16 x 98
        V = np.zeros((m * p, rank_total), dtype=float) # 32 x 98
        W = np.zeros((p * n, rank_total), dtype=float) # 32 x 98
        
        # Left half (k in 0..3)
        U[:, 0:49] = U4
        # V4 is (16, 49) for 4x4x4. V4[j*4+k, c]
        # We need V[j*8+k, c] for k in 0..3
        for j in range(4):
            for k in range(4):
                V[j*8 + k, 0:49] = V4[j*4 + k, :]
        # W4 is (16, 49) for 4x4x4. W4[k*4+i, c]
        # We need W[k*8+i, c] for k in 0..3
        for k in range(4):
            for i in range(4):
                W[k*8 + i, 0:49] = W4[k*4 + i, :]
        
        # Right half (k in 4..7)
        U[:, 49:98] = U4
        # V4[j*4+k, c] -> V[j*8+4+k, c]
        for j in range(4):
            for k in range(4):
                V[j*8 + 4 + k, 49:98] = V4[j*4 + k, :]
        # W4[k*4+i, c] -> W[(k+4)*8+i, c]
        for k in range(4):
            for i in range(4):
                W[(k+4)*8 + i, 49:98] = W4[k*4 + i, :]
        
        # Quick verification
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank_total):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-8:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    except Exception:
        pass
    
    # Fallback: Schoolbook (rank 128)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
