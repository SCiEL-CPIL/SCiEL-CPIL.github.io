import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i_idx, j_idx, k_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        if col >= rank:
            return
        # U index: i * m + j
        U[i_idx * m + j_idx][col] += u_val
        # V index: j * p + k
        V[j_idx * p + k_idx][col] += v_val
        # W index: k * n + i
        W[k_idx * n + i_idx][col] += w_val
        col += 1

    # Function to add Strassen terms for 2x2 block
    # i_off, j_off, k_off are offsets for i, j, k
    def add_strassen_2x2(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        add_term(i_off, j_off, k_off) # A00 B00
        add_term(i_off+1, j_off+1, k_off) # A11 B00 -- WAIT
        # Strassen combines them. We need to represent the linear combinations in U, V, W.
        # The rank 1 term is u \otimes v \otimes w.
        # For M1, u has 1s at (0,0) and (1,1). v has 1s at (0,0) and (1,1). w has 1s at (0,0) and (1,1) with signs?
        # No, the decomposition is a sum of rank 1 tensors.
        # M1 corresponds to one rank 1 tensor where:
        # u is vector with 1 at (0,0) and 1 at (1,1) (relative to block)
        # v is vector with 1 at (0,0) and 1 at (1,1)
        # w is vector with 1 at (0,0) and 1 at (1,1)
        # But wait, Strassen formula:
        # C11 = M1 + M4 - M5 + M7
        # This means the W vectors for M1, M4, M5, M7 will have coefficients in the reconstruction.
        # But the decomposition is T = sum u_k \otimes v_k \otimes w_k.
        # The output C is computed as sum (u_k \cdot A) (v_k \cdot B) w_k.
        # So w_k must be the basis vector for C? No, w_k can be any vector in C space.
        # In the standard Strassen decomposition:
        # M1 = (A11+A22)(B11+B22).
        # This contributes to C11, C12, C21, C22.
        # Specifically, C11 gets +M1, C12 gets 0? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # So for M1, the w vector is (1, 0, 0, 1) in (C11, C12, C21, C22) basis?
        # No, M1 contributes to C11 and C22.
        # So w_M1 has 1 at C11 and 1 at C22.
        
        # Let's define the 7 terms explicitly for the 2x2 block.
        # Indices relative to block: (0,0), (0,1), (1,0), (1,1) for A, B, C.
        
        # Term 1: M1
        # u: (0,0)=1, (1,1)=1
        # v: (0,0)=1, (1,1)=1
        # w: (0,0)=1, (1,1)=1  (C11, C22)
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # Term 2: M2 = (A21+A22)B11
        # u: (1,0)=1, (1,1)=1
        # v: (0,0)=1
        # w: (1,0)=1 (C21)
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # Term 3: M3 = A11(B21+B22)
        # u: (0,0)=1
        # v: (1,0)=1, (1,1)=1
        # w: (0,1)=1 (C12)
        U[i_off * m + j_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # Term 4: M4 = A22(B12-B22)
        # u: (1,1)=1
        # v: (0,1)=1, (1,1)=-1
        # w: (0,0)=1, (1,0)=1 (C11, C21)
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # Term 5: M5 = (A11+A12)B22
        # u: (0,0)=1, (0,1)=1
        # v: (1,1)=1
        # w: (0,1)=1, (1,1)=-1 (C12, C22) -- Wait, C22 gets -M5?
        # C22 = M1 - M2 + M3 + M6. No M5 in C22.
        # C12 = M3 + M5.
        # C11 = M1 + M4 - M5 + M7.
        # So M5 contributes to C12 (+) and C11 (-).
        # w: (0,1)=1, (0,0)=-1
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + i_off][col] = 1
        W[k_off * n + i_off][col] = -1
        col += 1
        
        # Term 6: M6 = (A22-A11)(B11+B12)
        # u: (1,1)=1, (0,0)=-1
        # v: (0,0)=1, (0,1)=1
        # w: (1,1)=1 (C22)
        U[(i_off+1) * m + (j_off+1)][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # Term 7: M7 = (A12-A22)(B21+B11)
        # u: (0,1)=1, (1,1)=-1
        # v: (1,0)=1, (0,0)=1
        # w: (0,0)=1 (C11)
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1

    # Function to add 4 terms for A_2x1 * B_1x2 (Outer product part)
    # i_off is row offset, j_idx is col index (fixed), k_off is col offset
    def add_outer_2x2(i_off, j_idx, k_off):
        nonlocal col
        # A is 2x1 at col j_idx. B is 1x2 at row j_idx.
        # C is 2x2.
        # C_00 = A_0 B_0
        # C_01 = A_0 B_1
        # C_10 = A_1 B_0
        # C_11 = A_1 B_1
        for di in range(2):
            for dk in range(2):
                U[(i_off + di) * m + j_idx][col] = 1
                V[j_idx * p + (k_off + dk)][col] = 1
                W[(k_off + dk) * n + (i_off + di)][col] = 1
                col += 1

    # Function to add 4 terms for A_1x2 * B_2x2
    # i_idx is row index (fixed), j_off is col offset, k_off is col offset
    def add_row_vec_mat(i_idx, j_off, k_off):
        nonlocal col
        # A is 1x2 at row i_idx. B is 2x2.
        # C is 1x2.
        # C_0 = A_0 B_00 + A_1 B_10
        # C_1 = A_0 B_01 + A_1 B_11
        for dj in range(2):
            for dk in range(2):
                U[i_idx * m + (j_off + dj)][col] = 1
                V[(j_off + dj) * p + (k_off + dk)][col] = 1
                W[(k_off + dk) * n + i_idx][col] = 1
                col += 1

    # Function to add 2 terms for A_1x1 * B_1x2
    def add_scalar_row(i_idx, j_idx, k_off):
        nonlocal col
        for dk in range(2):
            U[i_idx * m + j_idx][col] = 1
            V[j_idx * p + (k_off + dk)][col] = 1
            W[(k_off + dk) * n + i_idx][col] = 1
            col += 1

    # Block 1: k=0,1
    # i in 0,1 for Strassen
    add_strassen_2x2(0, 0, 0)
    # A_12 B_21: i in 0,1, j=2, k in 0,1
    add_outer_2x2(0, 2, 0)
    # A_21 B_11: i=2, j in 0,1, k in 0,1
    add_row_vec_mat(2, 0, 0)
    # A_22 B_21: i=2, j=2, k in 0,1
    add_scalar_row(2, 2, 0)

    # Block 2: k=2,3
    add_strassen_2x2(0, 0, 2)
    add_outer_2x2(0, 2, 2)
    add_row_vec_mat(2, 0, 2)
    add_scalar_row(2, 2, 2)

    return {"U": U, "V": V, "W": W}
