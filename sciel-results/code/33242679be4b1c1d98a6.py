import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # 1. Define 2x2x2 Strassen vectors
    # u: 4 entries (a,b,c,d)
    # v: 4 entries (e,f,g,h)
    # w: 4 entries (C11, C12, C21, C22)
    
    # M1 = (a+d)(e+h)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C11 + C22
    
    # M2 = (c+d)e
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # C21 - C22? 
    # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So M2 contributes +1 to C21, -1 to C22.
    # w2 = [0, 0, 1, -1] is correct.
    
    # M3 = a(f-h)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1] # C12 + C22
    
    # M4 = d(g-e)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0] # C11 + C21
    
    # M5 = (a+b)h
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # -C11 + C12
    
    # M6 = (c-a)(e+f)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # C22
    
    # M7 = (b-d)(g+h)
    u7 = [0, 1, -1, 0]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # C11
    
    strassen_vecs = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Strassen terms for 4x4x4
    # Each term is (A_blocks, B_blocks, C_blocks)
    # Blocks are (row_block, col_block, sign)
    
    # M1 = (A11 + A22)(B11 + B22) -> C11 + C22
    term1 = (
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)]
    )
    
    # M2 = (A21 + A22)B11 -> C21 - C22
    term2 = (
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)]
    )
    
    # M3 = A11(B12 - B22) -> C12 + C22
    term3 = (
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(0,1,1), (1,1,1)]
    )
    
    # M4 = A22(B21 - B11) -> C11 + C21
    term4 = (
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (1,0,1)]
    )
    
    # M5 = (A11 + A12)B22 -> -C11 + C12
    term5 = (
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(0,0,-1), (0,1,1)]
    )
    
    # M6 = (A21 - A11)(B11 + B12) -> C22
    term6 = (
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,1,1)]
    )
    
    # M7 = (A12 - A22)(B21 + B22) -> C11
    term7 = (
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)]
    )
    
    strassen_terms = [term1, term2, term3, term4, term5, term6, term7]
    
    # Helper to get index in U, V, W
    # U index: i*m + j. i in 0..3, j in 0..3.
    # Block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1.
    # Entry (r, c) in block (br, bc) is at global (2*br+r, 2*bc+c).
    
    def get_u_index(br, bc, r, c):
        gi = 2*br + r
        gj = 2*bc + c
        return gi * m + gj
        
    def get_v_index(br, bc, r, c, col_offset):
        # V index: j*p + k. j is row of B (0..3), k is col of B (0..8).
        # Block (br, bc) of B in the 4x4 part.
        # But B is 4x9.
        # For the first 4x4 part, cols are 0..3.
        # For the second 4x4 part, cols are 4..7.
        # Block (br, bc) in the 4x4 grid of B.
        # Row j = 2*br + r.
        # Col k = col_offset + 2*bc + c.
        gj = 2*br + r
        gk = col_offset + 2*bc + c
        return gj * p + gk
        
    def get_w_index(br, bc, r, c, col_offset):
        # W index: k*n + i. k is col of C (0..8), i is row of C (0..3).
        # Block (br, bc) of C.
        # Row i = 2*br + r.
        # Col k = col_offset + 2*bc + c.
        gi = 2*br + r
        gk = col_offset + 2*bc + c
        return gk * n + gi

    # Build decomposition
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_vector(u_vec, v_vec, w_vec):
        nonlocal rank
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        rank += 1
        
    # Process 4x4x4 part for cols 0-3
    for term in strassen_terms:
        a_blocks, b_blocks, c_blocks = term
        for local_u, local_v, local_w in strassen_vecs:
            u_global = [0] * (n*m)
            v_global = [0] * (m*p)
            w_global = [0] * (p*n)
            
            # Fill u_global
            for br, bc, sign in a_blocks:
                for r in range(2):
                    for c in range(2):
                        val = local_u[r*2+c]
                        if val != 0:
                            idx = get_u_index(br, bc, r, c)
                            u_global[idx] += sign * val
                            
            # Fill v_global
            for br, bc, sign in b_blocks:
                for r in range(2):
                    for c in range(2):
                        val = local_v[r*2+c]
                        if val != 0:
                            idx = get_v_index(br, bc, r, c, 0)
                            v_global[idx] += sign * val
                            
            # Fill w_global
            for br, bc, sign in c_blocks:
                for r in range(2):
                    for c in range(2):
                        val = local_w[r*2+c]
                        if val != 0:
                            idx = get_w_index(br, bc, r, c, 0)
                            w_global[idx] += sign * val
                            
            add_vector(u_global, v_global, w_global)
            
    # Process 4x4x4 part for cols 4-7
    for term in strassen_terms:
        a_blocks, b_blocks, c_blocks = term
        for local_u, local_v, local_w in strassen_vecs:
            u_global = [0] * (n*m)
            v_global = [0] * (m*p)
            w_global = [0] * (p*n)
            
            # Fill u_global (same A)
            for br, bc, sign in a_blocks:
                for r in range(2):
                    for c in range(2):
                        val = local_u[r*2+c]
                        if val != 0:
                            idx = get_u_index(br, bc, r, c)
                            u_global[idx] += sign * val
                            
            # Fill v_global (B cols 4-7)
            for br, bc, sign in b_blocks:
                for r in range(2):
                    for c in range(2):
                        val = local_v[r*2+c]
                        if val != 0:
                            idx = get_v_index(br, bc, r, c, 4)
                            v_global[idx] += sign * val
                            
            # Fill w_global (C cols 4-7)
            for br, bc, sign in c_blocks:
                for r in range(2):
                    for c in range(2):
                        val = local_w[r*2+c]
                        if val != 0:
                            idx = get_w_index(br, bc, r, c, 4)
                            w_global[idx] += sign * val
                            
            add_vector(u_global, v_global, w_global)
            
    # Process 4x4x1 part for col 8
    # C[i,8] = sum_j A[i,j] B[j,8]
    for i in range(n):
        for j in range(m):
            u_global = [0] * (n*m)
            v_global = [0] * (m*p)
            w_global = [0] * (p*n)
            
            u_idx = i * m + j
            u_global[u_idx] = 1
            
            v_idx = j * p + 8
            v_global[v_idx] = 1
            
            w_idx = 8 * n + i
            w_global[w_idx] = 1
            
            add_vector(u_global, v_global, w_global)
            
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U, "V": V, "W": W}
