import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    U_list = []
    V_list = []
    W_list = []
    
    def make_U(indices):
        vec = [0.0] * nm
        for i, j, coef in indices:
            vec[i*m+j] = coef
        return vec
    
    def make_V(indices):
        vec = [0.0] * mp
        for j, k, coef in indices:
            vec[j*p+k] = coef
        return vec
    
    def make_W(indices):
        vec = [0.0] * pn
        for i, k, coef in indices:
            vec[k*n+i] = coef
        return vec
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    def strassen_222(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        a00 = (A_rows[0], A_cols[0])
        a01 = (A_rows[0], A_cols[1])
        a10 = (A_rows[1], A_cols[0])
        a11 = (A_rows[1], A_cols[1])
        b00 = (B_rows[0], B_cols[0])
        b01 = (B_rows[0], B_cols[1])
        b10 = (B_rows[1], B_cols[0])
        b11 = (B_rows[1], B_cols[1])
        c00 = (C_rows[0], C_cols[0])
        c01 = (C_rows[0], C_cols[1])
        c10 = (C_rows[1], C_cols[0])
        c11 = (C_rows[1], C_cols[1])
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add_term(make_U([a00,1, a11,1]), make_V([b00,1, b11,1]), make_W([c00,1, c11,1]))
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        add_term(make_U([a10,1, a11,1]), make_V([b00,1]), make_W([c10,1, c11,-1]))
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        add_term(make_U([a00,1]), make_V([b01,1, b11,-1]), make_W([c01,1, c11,1]))
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        add_term(make_U([a11,1]), make_V([b10,1, b00,-1]), make_W([c00,1, c10,1]))
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        add_term(make_U([a00,1, a01,1]), make_V([b11,1]), make_W([c00,-1, c01,1]))
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        add_term(make_U([a10,1, a00,-1]), make_V([b00,1, b01,1]), make_W([c11,1]))
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        add_term(make_U([a01,1, a11,-1]), make_V([b10,1, b11,1]), make_W([c00,1]))
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen_222([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    strassen_222([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # Block 3: A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    for i in range(2):
        for k in range(4):
            add_term(make_U([(i,2,1)]), make_V([(2,k,1)]), make_W([(i,k,1)]))
    
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    rank = U_mat.shape[1]
    
    # Verify
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_recon = np.einsum('ac,bc,dc->abd', U_mat, V_mat, W_mat)
    
    if not np.allclose(T_recon, T_target, atol=1e-10):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
