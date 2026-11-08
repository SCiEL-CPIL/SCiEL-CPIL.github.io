import time
import numpy as np

_DIM = (6, 6, 6)

def _residual_max(U, V, W, n, m, p):
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    R_3d = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    residual = np.max(np.abs(R_3d - 1.0))
    return float(residual)

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Row indexing:
    U row = i*2 + j  (A_{i,j})
    V row = j*2 + k  (B_{j,k})
    W row = k*2 + i  (C_{k,i})
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # M3 = A11(B21 - B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[0, 2] = 1
    
    # M4 = A22(B12 - B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def construct():
    n, m, p = _DIM
    deadline = time.time() + 115.0  # Safety margin

    U_s, V_s, W_s = _schoolbook(n, m, p)
    best = (U_s.copy(), V_s.copy(), W_s.copy())
    best_rank = U_s.shape[1]
    
    # Try Strassen tensor product decomposition
    # This corresponds to 3x3 grid of 2x2 blocks.
    # U_strassen (4x7), V_strassen (4x7), W_strassen (4x7)
    # U_3x3 (9x27), V_3x3 (9x27), W_3x3 (9x27)
    
    if time.time() < deadline:
        U2, V2, W2 = _strassen_2x2()
        U3, V3, W3 = _schoolbook(3, 3, 3)
        
        rank_strassen = 7 * 27 # 189
        U_new = np.zeros((36, rank_strassen))
        V_new = np.zeros((36, rank_strassen))
        W_new = np.zeros((36, rank_strassen))
        
        # Indices for U: row = i * m + j = (3*i1 + i2) * 6 + (3*j1 + j2) = 18*i1 + 6*i2 + 3*j1 + j2
        # Col = 27*r1 + r2 where r1 in 0..6, r2 in 0..26
        for i1 in range(2):
            for i2 in range(3):
                for j1 in range(2):
                    for j2 in range(3):
                        row_u = (3 * i1 + i2) * 6 + (3 * j1 + j2)
                        
                        for r1 in range(7):
                            u2_val = U2[i1 * 2 + j1, r1]
                            if u2_val == 0: continue
                            
                            for r2 in range(27):
                                u3_val = U3[i2 * 3 + j2, r2]
                                if u3_val == 0: continue
                                
                                col = r1 * 27 + r2
                                U_new[row_u, col] = u2_val * u3_val
        
        # Indices for V: row = j * p + k = (3*j1 + j2) * 6 + (3*k1 + k2) = 18*j1 + 6*j2 + 3*k1 + k2
        for j1 in range(2):
            for j2 in range(3):
                for k1 in range(2):
                    for k2 in range(3):
                        row_v = (3 * j1 + j2) * 6 + (3 * k1 + k2)
                        
                        for r1 in range(7):
                            v2_val = V2[j1 * 2 + k1, r1]
                            if v2_val == 0: continue
                            
                            for r2 in range(27):
                                v3_val = V3[j2 * 3 + k2, r2]
                                if v3_val == 0: continue
                                
                                col = r1 * 27 + r2
                                V_new[row_v, col] = v2_val * v3_val
        
        # Indices for W: row = k * n + i = (3*k1 + k2) * 6 + (3*i1 + i2) = 18*k1 + 6*k2 + 3*i1 + i2
        for k1 in range(2):
            for k2 in range(3):
                for i1 in range(2):
                    for i2 in range(3):
                        row_w = (3 * k1 + k2) * 6 + (3 * i1 + i2)
                        
                        for r1 in range(7):
                            w2_val = W2[k1 * 2 + i1, r1]
                            if w2_val == 0: continue
                            
                            for r2 in range(27):
                                w3_val = W3[k2 * 3 + i2, r2]
                                if w3_val == 0: continue
                                
                                col = r1 * 27 + r2
                                W_new[row_w, col] = w2_val * w3_val
        
        # Verify
        res = _residual_max(U_new, V_new, W_new, n, m, p)
        if res < 1e-7:
            best = (U_new.copy(), V_new.copy(), W_new.copy())
            best_rank = rank_strassen

    Uf, Vf, Wf = best
    Uo = [[round(float(v), 9) for v in row] for row in Uf]
    Vo = [[round(float(v), 9) for v in row] for row in Vf]
    Wo = [[round(float(v), 9) for v in row] for row in Wf]
    return {"U": Uo, "V": Vo, "W": Wo}
