import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Direct Strassen for 4x4 * 4x4 (rank 49)
    # Split into 2x2 blocks of 2x2 matrices
    # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]], C = [[C11,C12],[C21,C22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2*2x2 product, decomposed with Strassen (rank 7 each)
    # Total: 7*7 = 49 rank-1 terms
    
    strassen_2x2 = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    # Block structure for 4x4*4x4
    # A blocks: (row, col) -> 2x2 submatrix
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # For each of the 7 Strassen multipliers, define:
    # - Which A blocks contribute and with what coefficients
    # - Which B blocks contribute and with what coefficients
    # - Which C blocks receive the result and with what coefficients
    
    block_defs = [
        # M1 = (A11+A22)(B11+B22) -> C11:+1, C22:+1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1, (1,1):1}),
        # M2 = (A21+A22)B11 -> C21:+1, C22:-1
        ([(1,0,1),(1,1,1)], [(0,0,1)], {(1,0):1, (1,1):-1}),
        # M3 = A11(B12-B22) -> C12:+1, C22:+1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], {(0,1):1, (1,1):1}),
        # M4 = A22(B21-B11) -> C11:+1, C21:+1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], {(0,0):1, (1,0):1}),
        # M5 = (A11+A12)B22 -> C11:-1, C12:+1
        ([(0,0,1),(0,1,1)], [(1,1,1)], {(0,0):-1, (0,1):1}),
        # M6 = (A21-A11)(B11+B12) -> C22:+1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], {(1,1):1}),
        # M7 = (A12-A22)(B21+B22) -> C11:+1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], {(0,0):1}),
    ]
    
    def add_strassen_4x4(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k):
        """Add 49 rank-1 terms for 4x4*4x4 with given offsets."""
        for A_blocks, B_blocks, C_blocks in block_defs:
            for A_meta, B_meta, C_meta in strassen_2x2:
                u = [0]*16
                v = [0]*36
                w = [0]*36
                
                # Build u: A side
                for li_lj, coeff in A_meta.items():
                    li = li_lj // 2
                    lj = li_lj % 2
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = a_off_i + br*2 + li
                        gj = a_off_j + bc*2 + lj
                        u[gi*m + gj] += coeff * bc_coeff
                
                # Build v: B side
                for lj_lk, coeff in B_meta.items():
                    lj = lj_lk // 2
                    lk = lj_lk % 2
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = b_off_j + br*2 + lj
                        gk = b_off_k + bc*2 + lk
                        v[gj*p + gk] += coeff * bc_coeff
                
                # Build w: C side
                for li_lk, coeff in C_meta.items():
                    li = li_lk // 2
                    lk = li_lk % 2
                    for (br, bc), bc_coeff in C_blocks.items():
                        gi = c_off_i + br*2 + li
                        gk = c_off_k + bc*2 + lk
                        w[gk*n + gi] += coeff * bc_coeff
                
                add_term(u, v, w)
    
    # 4x4*4 for columns 0-3
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    # 4x4*4 for columns 4-7
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    # 4x4*1 for column 8 (schoolbook, rank 16)
    for i in range(4):
        for j in range(4):
            u = [0]*16
            v = [0]*36
            w = [0]*36
            u[i*m + j] = 1
            v[j*p + 8] = 1
            w[8*n + i] = 1
            add_term(u, v, w)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}
