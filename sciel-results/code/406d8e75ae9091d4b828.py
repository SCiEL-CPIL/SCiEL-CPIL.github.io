import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 140  # Safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if i==i' and j==j' and k==k'
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact"""
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(result, T, atol=tol)

    def add_strassen_block(U_list, V_list, W_list, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen rank-7 decomposition for a 2x2x2 block.
        a_rows, a_cols: global row/col indices for A block (2x2)
        b_rows, b_cols: global row/col indices for B block (2x2)
        c_rows, c_cols: global row/col indices for C block (2x2)
        """
        # Local indices to global indices
        # A local (0,0) -> global (a_rows[0], a_cols[0])
        # A local (0,1) -> global (a_rows[0], a_cols[1])
        # A local (1,0) -> global (a_rows[1], a_cols[0])
        # A local (1,1) -> global (a_rows[1], a_cols[1])
        
        def get_a_idx(lr, lc):
            return a_rows[lr] * m + a_cols[lc]
            
        def get_b_idx(lr, lc):
            return b_rows[lr] * p + b_cols[lc]
            
        def get_c_idx(lr, lc):
            return c_cols[lc] * n + c_rows[lr]

        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        terms = []
        
        # M1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(0,0)] = 1; u[get_a_idx(1,1)] = 1
        v[get_b_idx(0,0)] = 1; v[get_b_idx(1,1)] = 1
        w[get_c_idx(0,0)] = 1; w[get_c_idx(1,1)] = 1
        terms.append((u, v, w))
        
        # M2
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(1,0)] = 1; u[get_a_idx(1,1)] = 1
        v[get_b_idx(0,0)] = 1
        w[get_c_idx(1,0)] = 1; w[get_c_idx(1,1)] = -1 # Wait, C21 = M2+M4, C22 = M1-M2+...
        # M2 contributes to C21 (+1) and C22 (-1)
        w[get_c_idx(1,0)] = 1
        w[get_c_idx(1,1)] = -1
        terms.append((u, v, w))
        
        # M3
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(0,0)] = 1
        v[get_b_idx(1,0)] = 1; v[get_b_idx(1,1)] = -1
        w[get_c_idx(0,1)] = 1; w[get_c_idx(1,1)] = 1
        terms.append((u, v, w))
        
        # M4
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(1,1)] = 1
        v[get_b_idx(0,1)] = 1; v[get_b_idx(0,0)] = -1
        w[get_c_idx(0,0)] = 1; w[get_c_idx(1,0)] = 1
        terms.append((u, v, w))
        
        # M5
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(0,0)] = 1; u[get_a_idx(0,1)] = 1
        v[get_b_idx(1,1)] = 1
        w[get_c_idx(0,0)] = -1; w[get_c_idx(0,1)] = 1
        terms.append((u, v, w))
        
        # M6
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(1,0)] = 1; u[get_a_idx(0,0)] = -1
        v[get_b_idx(0,0)] = 1; v[get_b_idx(0,1)] = 1
        w[get_c_idx(1,1)] = 1
        terms.append((u, v, w))
        
        # M7
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(0,1)] = 1; u[get_a_idx(1,1)] = -1
        v[get_b_idx(1,0)] = 1; v[get_b_idx(1,1)] = 1
        w[get_c_idx(0,0)] = 1; w[get_c_idx(1,1)] = 1
        terms.append((u, v, w))
        
        for u, v, w in terms:
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)

    def add_schoolbook_1x2x2_block(U_list, V_list, W_list, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add schoolbook rank-4 decomposition for a 1x2x2 block.
        A is 1x2, B is 2x2, C is 1x2.
        """
        def get_a_idx(lr, lc):
            return a_rows[lr] * m + a_cols[lc]
            
        def get_b_idx(lr, lc):
            return b_rows[lr] * p + b_cols[lc]
            
        def get_c_idx(lr, lc):
            return c_cols[lc] * n + c_rows[lr]

        # Terms:
        # T1 = A1 B11 -> C1
        # T2 = A1 B12 -> C2
        # T3 = A2 B21 -> C1
        # T4 = A2 B22 -> C2
        
        # T1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(0,0)] = 1
        v[get_b_idx(0,0)] = 1
        w[get_c_idx(0,0)] = 1
        U_list.append(u); V_list.append(v); W_list.append(w)
        
        # T2
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(0,0)] = 1
        v[get_b_idx(0,1)] = 1
        w[get_c_idx(0,1)] = 1
        U_list.append(u); V_list.append(v); W_list.append(w)
        
        # T3
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(0,1)] = 1
        v[get_b_idx(1,0)] = 1
        w[get_c_idx(0,0)] = 1
        U_list.append(u); V_list.append(v); W_list.append(w)
        
        # T4
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[get_a_idx(0,1)] = 1
        v[get_b_idx(1,1)] = 1
        w[get_c_idx(0,1)] = 1
        U_list.append(u); V_list.append(v); W_list.append(w)

    U_list = []
    V_list = []
    W_list = []
    
    # 8 2x2x2 blocks
    # A rows 0,1. A cols 0,1 and 2,3.
    # B rows 0,1 and 2,3. B cols 0,1; 2,3; 4,5; 6,7.
    
    # Block 1: A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(U_list, V_list, W_list, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # Block 2: A[0:2, 0:2], B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(U_list, V_list, W_list, [0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # Block 3: A[0:2, 0:2], B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_block(U_list, V_list, W_list, [0,1], [0,1], [0,1], [4,5], [0,1], [4,5])
    # Block 4: A[0:2, 0:2], B[0:2, 6:8] -> C[0:2, 6:8]
    add_strassen_block(U_list, V_list, W_list, [0,1], [0,1], [0,1], [6,7], [0,1], [6,7])
    
    # Block 5: A[0:2, 2:4], B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(U_list, V_list, W_list, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    # Block 6: A[0:2, 2:4], B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(U_list, V_list, W_list, [0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    # Block 7: A[0:2, 2:4], B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen_block(U_list, V_list, W_list, [0,1], [2,3], [2,3], [4,5], [0,1], [4,5])
    # Block 8: A[0:2, 2:4], B[2:4, 6:8] -> C[0:2, 6:8]
    add_strassen_block(U_list, V_list, W_list, [0,1], [2,3], [2,3], [6,7], [0,1], [6,7])
    
    # 8 1x2x2 blocks
    # A row 2. A cols 0,1 and 2,3.
    # B rows 0,1 and 2,3. B cols 0,1; 2,3; 4,5; 6,7.
    
    # Block 9: A[2:3, 0:2], B[0:2, 0:2] -> C[2:3, 0:2]
    add_schoolbook_1x2x2_block(U_list, V_list, W_list, [2], [0,1], [0,1], [0,1], [2], [0,1])
    # Block 10: A[2:3, 0:2], B[0:2, 2:4] -> C[2:3, 2:4]
    add_schoolbook_1x2x2_block(U_list, V_list, W_list, [2], [0,1], [0,1], [2,3], [2], [2,3])
    # Block 11: A[2:3, 0:2], B[0:2, 4:6] -> C[2:3, 4:6]
    add_schoolbook_1x2x2_block(U_list, V_list, W_list, [2], [0,1], [0,1], [4,5], [2], [4,5])
    # Block 12: A[2:3, 0:2], B[0:2, 6:8] -> C[2:3, 6:8]
    add_schoolbook_1x2x2_block(U_list, V_list, W_list, [2], [0,1], [0,1], [6,7], [2], [6,7])
    
    # Block 13: A[2:3, 2:4], B[2:4, 0:2] -> C[2:3, 0:2]
    add_schoolbook_1x2x2_block(U_list, V_list, W_list, [2], [2,3], [2,3], [0,1], [2], [0,1])
    # Block 14: A[2:3, 2:4], B[2:4, 2:4] -> C[2:3, 2:4]
    add_schoolbook_1x2x2_block(U_list, V_list, W_list, [2], [2,3], [2,3], [2,3], [2], [2,3])
    # Block 15: A[2:3, 2:4], B[2:4, 4:6] -> C[2:3, 4:6]
    add_schoolbook_1x2x2_block(U_list, V_list, W_list, [2], [2,3], [2,3], [4,5], [2], [4,5])
    # Block 16: A[2:3, 2:4], B[2:4, 6:8] -> C[2:3, 6:8]
    add_schoolbook_1x2x2_block(U_list, V_list, W_list, [2], [2,3], [2,3], [6,7], [2], [6,7])
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    if verify(U, V, W):
        best_U, best_V, best_W, best_rank = U, V, W, U.shape[1]
        
    # Convert to list format
    U_out = [[U_out_val for U_out_val in row] for row in best_U.tolist()]
    V_out = [[V_out_val for V_out_val in row] for row in best_V.tolist()]
    W_out = [[W_out_val for W_out_val in row] for row in best_W.tolist()]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
