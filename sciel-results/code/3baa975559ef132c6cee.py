import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2 Strassen U, V, W
    # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U22 = [[0]*7 for _ in range(4)]
    V22 = [[0]*7 for _ in range(4)]
    W22 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U22[0][0] = 1; U22[3][0] = 1
    V22[0][0] = 1; V22[3][0] = 1
    W22[0][0] = 1; W22[3][0] = 1; W22[2][0] = -1; W22[1][0] = 1 # C00+=M1, C11+=M1
    # Wait, W convention is W[k*n+i, c]. For 2x2, W[k*2+i, c].
    # C00 is at [0,0] -> index 0. C11 is at [1,1] -> index 3.
    # C01 is at [0,1] -> index 1. C10 is at [1,0] -> index 2.
    # Let's re-verify W indices for 2x2.
    # C[i,k]. Index k*2+i.
    # C00: k=0, i=0 -> 0.
    # C01: k=1, i=0 -> 2.
    # C10: k=0, i=1 -> 1.
    # C11: k=1, i=1 -> 3.
    
    # Reset W22
    W22 = [[0]*7 for _ in range(4)]
    
    # M1
    U22[0][0] = 1; U22[3][0] = 1
    V22[0][0] = 1; V22[3][0] = 1
    W22[0][0] = 1 # C00
    W22[3][0] = 1 # C11
    
    # M2 = (A10+A11)B00
    U22[2][1] = 1; U22[3][1] = 1
    V22[0][1] = 1
    W22[1][1] = 1 # C10
    W22[3][1] = -1 # C11
    
    # M3 = A00(B10-B11)
    U22[0][2] = 1
    V22[2][2] = 1; V22[3][2] = -1
    W22[2][2] = 1 # C01
    W22[3][2] = 1 # C11
    
    # M4 = A11(B01-B00)
    U22[3][3] = 1
    V22[1][3] = 1; V22[0][3] = -1
    W22[0][3] = 1 # C00
    W22[1][3] = 1 # C10
    
    # M5 = (A00+A01)B11
    U22[0][4] = 1; U22[1][4] = 1
    V22[3][4] = 1
    W22[0][4] = -1 # C00
    W22[2][4] = 1 # C01
    
    # M6 = (A10-A00)(B00+B01)
    U22[2][5] = 1; U22[0][5] = -1
    V22[0][5] = 1; V22[1][5] = 1
    W22[3][5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U22[1][6] = 1; U22[3][6] = -1
    V22[2][6] = 1; V22[3][6] = 1
    W22[0][6] = 1 # C00
    W22[2][6] = 1 # C01
    
    # 2. Construct 4x4 Strassen U, V, W via Kronecker Product
    # U44 = U22 (x) U22
    # Size: (4*4, 7*7) = (16, 49)
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    idx_A = (i1*2 + i2)*4 + (j1*2 + j2)
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1*7 + c2
                            U44[idx_A][c] = U22[i1*2+j1][c1] * U22[i2*2+j2][c2]
                            
                            idx_B = (j1*2 + j2)*4 + (j1*2 + j2) # Wait, B index is j,k.
                            # In 2x2, V index is j*2+k.
                            # In 4x4, V index is J*4+K. J=2*j1+j2, K=2*j2+k2?
                            # No, B is 4x4. Indices J, K in 0..3.
                            # J = 2*j1 + j2, K = 2*k1 + k2.
                            # So V44 index is (2*j1+j2)*4 + (2*k1+k2).
                            # V22 index for term (j1, k1) is j1*2+k1.
                            # V22 index for term (j2, k2) is j2*2+k2.
                            # So V44[idx_B][c] = V22[j1*2+k1][c1] * V22[j2*2+k2][c2].
                            
                            # Let's rewrite the loop properly.
                            
    # Redo Kronecker Product Construction
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            # A index: I = 2*i1+i2, J = 2*j1+j2 -> I*4+J
                            idx_A = (2*i1+i2)*4 + (2*j1+j2)
                            # B index: J = 2*j1+j2, K = 2*k1+k2 -> J*4+K
                            idx_B = (2*j1+j2)*4 + (2*k1+k2)
                            # C index: K = 2*k1+k2, I = 2*i1+i2 -> K*4+I
                            idx_C = (2*k1+k2)*4 + (2*i1+i2)
                            
                            for c1 in range(7):
                                for c2 in range(7):
                                    c = c1*7 + c2
                                    u_val = U22[i1*2+j1][c1] * U22[i2*2+j2][c2]
                                    v_val = V22[j1*2+k1][c1] * V22[j2*2+k2][c2]
                                    w_val = W22[k1*2+i1][c1] * W22[k2*2+i2][c2]
                                    
                                    U44[idx_A][c] += u_val
                                    V44[idx_B][c] += v_val
                                    W44[idx_C][c] += w_val

    # 3. Construct Boundary Terms
    # Total Rank so far: 49.
    # We will append to U, V, W.
    
    U_list = [row[:] for row in U44]
    V_list = [row[:] for row in V44]
    W_list = [row[:] for row in W44]
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx, v_idx, w_idx are tuples of (row, col) in the 5x5 grid?
        # No, they are linear indices.
        # We need to add a new column.
        new_col_u = [0]*25
        new_col_v = [0]*25
        new_col_w = [0]*25
        new_col_u[u_idx] = val
        new_col_v[v_idx] = val
        new_col_w[w_idx] = val
        
        for r in range(25):
            U_list[r].append(new_col_u[r])
            V_list[r].append(new_col_v[r])
            W_list[r].append(new_col_w[r])

    # Term 2: A00 b (4x4 * 4x1)
    # C[i, 4] = sum_j A[i, j] b[j]
    # i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            u_idx = i*5 + j
            v_idx = j*5 + 4
            w_idx = 4*5 + i
            add_term(u_idx, v_idx, w_idx)

    # Term 3: a^T B00 (1x4 * 4x4)
    # C[4, k] = sum_j a[j] B[j, k]
    # j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            u_idx = 4*5 + j
            v_idx = j*5 + k
            w_idx = 4*5 + k
            add_term(u_idx, v_idx, w_idx)

    # Term 4: a^T b (1x4 * 4x1)
    # C[4, 4] = sum_j a[j] b[j]
    # j in 0..3
    for j in range(4):
        u_idx = 4*5 + j
        v_idx = j*5 + 4
        w_idx = 4*5 + 4
        add_term(u_idx, v_idx, w_idx)

    # Term 5: a b^T (4x1 * 1x4)
    # C[i, k] = a[i] b^T[k]
    # i in 0..3, k in 0..3
    for i in range(4):
        for k in range(4):
            u_idx = i*5 + 4
            v_idx = 4*5 + k
            w_idx = k*5 + i
            add_term(u_idx, v_idx, w_idx)

    # Term 6: a beta (4x1 * 1x1)
    # C[i, 4] = a[i] beta
    # i in 0..3
    for i in range(4):
        u_idx = i*5 + 4
        v_idx = 4*5 + 4
        w_idx = 4*5 + i
        add_term(u_idx, v_idx, w_idx)

    # Term 7: alpha b^T (1x1 * 1x4)
    # C[4, k] = alpha b^T[k]
    # k in 0..3
    for k in range(4):
        u_idx = 4*5 + 4
        v_idx = 4*5 + k
        w_idx = 4*5 + k
        add_term(u_idx, v_idx, w_idx)

    # Term 8: alpha beta (1x1 * 1x1)
    # C[4, 4] = alpha beta
    u_idx = 4*5 + 4
    v_idx = 4*5 + 4
    w_idx = 4*5 + 4
    add_term(u_idx, v_idx, w_idx)

    return {"U": U_list, "V": V_list, "W": W_list}
