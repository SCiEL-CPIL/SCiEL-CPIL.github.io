import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 2, 2, 4
    rank = 14
    
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    
    # Block 1 (k=0,1)
    # M1: (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[5,0] = 1
    # M2: (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    # M3: A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[5,2] = -1
    # M4: A11(B10-B00)
    U[3,3] = 1
    V[4,3] = 1; V[0,3] = -1
    # M5: (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[5,4] = 1
    # M6: (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    # M7: (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[4,6] = 1; V[5,6] = 1
    
    # Block 2 (k=2,3)
    # M8: (A00+A11)(B02+B13)
    U[0,7] = 1; U[3,7] = 1
    V[2,7] = 1; V[7,7] = 1
    # M9: (A10+A11)B02
    U[2,8] = 1; U[3,8] = 1
    V[2,8] = 1
    # M10: A00(B03-B13)
    U[0,9] = 1
    V[3,9] = 1; V[7,9] = -1
    # M11: A11(B12-B02)
    U[3,10] = 1
    V[6,10] = 1; V[2,10] = -1
    # M12: (A00+A01)B13
    U[0,11] = 1; U[1,11] = 1
    V[7,11] = 1
    # M13: (A10-A00)(B02+B03)
    U[2,12] = 1; U[0,12] = -1
    V[2,12] = 1; V[3,12] = 1
    # M14: (A01-A11)(B12+B13)
    U[1,13] = 1; U[3,13] = -1
    V[6,13] = 1; V[7,13] = 1
    
    # Solve for W: for all (i,j,k): sum_c U[2i+j,c]*V[4j+k,c]*W[2k+i,c] = 1
    # Variables: W[2k+i, c] for k=0..3, i=0..1, c=0..13 → 8*14=112 vars
    # Equations: 16
    
    A_mat = np.zeros((16, 112))
    b_vec = np.ones(16)
    
    eq_idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                for c in range(rank):
                    var_idx = (2*k + i) * rank + c
                    A_mat[eq_idx, var_idx] = U[2*i+j, c] * V[4*j+k, c]
                eq_idx += 1
    
    W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    W = W_flat.reshape(8, rank)
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(4):
                val = np.sum(U[2*i+j, :] * V[4*j+k, :] * W[2*k+i, :])
                if abs(val - 1) > 1e-10:
                    # Fallback to schoolbook
                    return _schoolbook()
    
    # Check if values are close to integer/half-integer
    # Round to nearest 0.5
    W_rounded = np.round(W * 2) / 2
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    
    # Verify rounded version
    ok = True
    for i in range(2):
        for j in range(2):
            for k in range(4):
                val = np.sum(U_rounded[2*i+j, :] * V_rounded[4*j+k, :] * W_rounded[2*k+i, :])
                if abs(val - 1) > 1e-10:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if ok:
        U, V, W = U_rounded, V_rounded, W_rounded
    
    if time.time() > deadline - 5:
        return _schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook():
    n, m, p = 2, 2, 4
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
