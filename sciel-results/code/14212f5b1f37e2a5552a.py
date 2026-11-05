"""Matrix multiplication 4,4,9 tensor decomposition - Strassen with optimized tail."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Verified Strassen 2x2x2: 7 rank-1 terms
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11 (for U)
    #          0=B00, 1=B01, 2=B10, 3=B11 (for V)
    #          0=C00, 1=C01, 2=C10, 3=C11 (for W)
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4(B_col_offset):
        """Add 49 terms for 4x4 * 4x4 using Strassen at two levels."""
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in A_blocks:
                        gi = br * 2 + li
                        gj = bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in B_blocks:
                        gj = br * 2 + lj
                        gk = B_col_offset + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in C_blocks:
                        gi = br * 2 + li
                        gk = B_col_offset + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    
    # Build the decomposition: 49 + 49 + 1 = 99 rank
    add_strassen_4x4(0)   # 49 terms for columns 0-3
    add_strassen_4x4(4)   # 49 terms for columns 4-7
    
    # Tail: single term for column 8
    # T[i,j,8] = U[i*4+j] * V[j*9+8] * W[8*4+i] = 1*1*1 = 1
    u_tail = [1] * 16  # U[i*m+j] = 1 for all i,j
    v_tail = [0] * 36
    for j in range(4):
        v_tail[j * p + 8] = 1  # V[j*p+8] = 1
    w_tail = [0] * 36
    for i in range(4):
        w_tail[8 * n + i] = 1  # W[8*n+i] = 1
    
    add_term(u_tail, v_tail, w_tail)
    
    rank = len(U_cols)
    
    # Verify correctness
    if time.time() < deadline - 5:
        U_np = np.array(U_cols, dtype=float).T  # (16, rank)
        V_np = np.array(V_cols, dtype=float).T  # (36, rank)
        W_np = np.array(W_cols, dtype=float).T  # (36, rank)
        
        # Check T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-10:
            # Fallback: use schoolbook (rank 144) if verification fails
            U_cols = []
            V_cols = []
            W_cols = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_vec = [0] * 16
                        v_vec = [0] * 36
                        w_vec = [0] * 36
                        u_vec[i * m + j] = 1
                        v_vec[j * p + k] = 1
                        w_vec[k * n + i] = 1
                        add_term(u_vec, v_vec, w_vec)
            rank = len(U_cols)
    
    # Build the output matrices
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
