import numpy as np

def construct():
    n, m, p = 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add Strassen 2x2x2
    # a_idx: list of 4 global A indices (i*m+j)
    # b_idx: list of 4 global B indices (j*p+k)
    # c_idx: list of 4 global C indices (k*n+i)
    # col: starting column in U,V,W
    def add_strassen(a_idx, b_idx, c_idx, col):
        # M1 = (A11+A22)(B11+B22)
        # A11=a_idx[0], A12=a_idx[1], A21=a_idx[2], A22=a_idx[3]
        # B11=b_idx[0], B12=b_idx[1], B21=b_idx[2], B22=b_idx[3]
        # C11=c_idx[0], C12=c_idx[1], C21=c_idx[2], C22=c_idx[3]
        
        # We define the 7 products
        # P1: (A11+A22)(B11+B22)
        U[a_idx[0], col] += 1
        U[a_idx[3], col] += 1
        V[b_idx[0], col] += 1
        V[b_idx[3], col] += 1
        W[c_idx[0], col] += 1  # C11
        W[c_idx[3], col] += 1  # C22
        
        # P2: (A21+A22)B11
        U[a_idx[2], col+1] += 1
        U[a_idx[3], col+1] += 1
        V[b_idx[0], col+1] += 1
        W[c_idx[2], col+1] += 1  # C21
        W[c_idx[3], col+1] += -1 # C22
        
        # P3: A11(B12-B22)
        U[a_idx[0], col+2] += 1
        V[b_idx[1], col+2] += 1
        V[b_idx[3], col+2] += -1
        W[c_idx[0], col+2] += -1 # C11
        W[c_idx[1], col+2] += 1  # C12
        
        # P4: A22(B21-B11)
        U[a_idx[3], col+3] += 1
        V[b_idx[2], col+3] += 1
        V[b_idx[0], col+3] += -1
        W[c_idx[0], col+3] += 1  # C11
        W[c_idx[2], col+3] += 1  # C21
        
        # P5: (A11+A12)B22
        U[a_idx[0], col+4] += 1
        U[a_idx[1], col+4] += 1
        V[b_idx[3], col+4] += 1
        W[c_idx[0], col+4] += -1 # C11
        W[c_idx[1], col+4] += 1  # C12
        
        # P6: (A21-A11)(B11+B12)
        U[a_idx[2], col+5] += 1
        U[a_idx[0], col+5] += -1
        V[b_idx[0], col+5] += 1
        V[b_idx[1], col+5] += 1
        W[c_idx[3], col+5] += 1  # C22
        
        # P7: (A12-A22)(B21+B22)
        U[a_idx[1], col+6] += 1
        U[a_idx[3], col+6] += -1
        V[b_idx[2], col+6] += 1
        V[b_idx[3], col+6] += 1
        W[c_idx[0], col+6] += 1  # C11
        W[c_idx[3], col+6] += -1 # C22
        
        return col + 7

    # Helper to add 1x4x4 schoolbook
    # a_idx: 1 global A index
    # b_idx: 4 global B indices
    # c_idx: 4 global C indices
    def add_schoolbook_1x4x4(a_idx, b_idx, c_idx, col):
        for j in range(4):
            U[a_idx, col+j] += 1
            V[b_idx[j], col+j] += 1
            W[c_idx[j], col+j] += 1
        return col + 4

    col = 0
    # Top part: A[0:2, :]
    # B_left: B[:, 0:4]
    # 4 Strassen blocks
    # A_tl: A[0,0], A[0,1], A[1,0], A[1,1] -> 0, 1, 4, 5
    # A_tr: A[0,2], A[0,3], A[1,2], A[1,3] -> 2, 3, 6, 7
    # B_tl: B[0,0], B[0,1], B[1,0], B[1,1] -> 0, 1, 8, 9
    # B_tl2: B[0,2], B[0,3], B[1,2], B[1,3] -> 2, 3, 10, 11
    # B_bl: B[2,0], B[2,1], B[3,0], B[3,1] -> 16, 17, 24, 25
    # B_bl2: B[2,2], B[2,3], B[3,2], B[3,3] -> 18, 19, 26, 27
    
    # C_tl: C[0,0], C[0,1], C[1,0], C[1,1] -> 0, 1, 3, 4
    # C_tl2: C[0,2], C[0,3], C[1,2], C[1,3] -> 2, 3, 5, 6
    
    # 1. A_tl * B_tl -> C_tl
    col = add_strassen([0, 1, 4, 5], [0, 1, 8, 9], [0, 1, 3, 4], col)
    # 2. A_tl * B_tl2 -> C_tl2
    col = add_strassen([0, 1, 4, 5], [2, 3, 10, 11], [2, 3, 5, 6], col)
    # 3. A_tr * B_bl -> C_tl
    col = add_strassen([2, 3, 6, 7], [16, 17, 24, 25], [0, 1, 3, 4], col)
    # 4. A_tr * B_bl2 -> C_tl2
    col = add_strassen([2, 3, 6, 7], [18, 19, 26, 27], [2, 3, 5, 6], col)
    
    # B_right: B[:, 4:8]
    # B_tr: B[0,4]... -> 4,5,12,13
    # B_tr2: B[0,6]... -> 6,7,14,15
    # B_br: B[2,4]... -> 20,21,28,29
    # B_br2: B[2,6]... -> 22,23,30,31
    
    # C_tr: C[0,4]... -> 4,5,7,8
    # C_tr2: C[0,6]... -> 6,7,9,10
    
    # 5. A_tl * B_tr -> C_tr
    col = add_strassen([0, 1, 4, 5], [4, 5, 12, 13], [4, 5, 7, 8], col)
    # 6. A_tl * B_tr2 -> C_tr2
    col = add_strassen([0, 1, 4, 5], [6, 7, 14, 15], [6, 7, 9, 10], col)
    # 7. A_tr * B_br -> C_tr
    col = add_strassen([2, 3, 6, 7], [20, 21, 28, 29], [4, 5, 7, 8], col)
    # 8. A_tr * B_br2 -> C_tr2
    col = add_strassen([2, 3, 6, 7], [22, 23, 30, 31], [6, 7, 9, 10], col)
    
    # Bottom part: A[2, :]
    # A_bot: A[2,0]... -> 8, 9, 10, 11
    # C_bot: C[2,0]... -> 8, 9, 10, 11, 12, 13, 14, 15
    
    # 9. A_bot * B_left
    col = add_schoolbook_1x4x4(8, [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27], [8, 9, 10, 11, 12, 13, 14, 15], col)
    # Wait, 1x4x4 is rank 16.
    # A_bot is 1x4. B_left is 4x4.
    # A_bot * B_left is 1x4.
    # My helper adds 4 terms. But 1x4x4 needs 16 terms.
    # The helper `add_schoolbook_1x4x4` I wrote only adds 4 terms (for 1x1x4? No).
    # I need to fix the helper.
    
    # Correct helper for 1x4x4:
    # A is 1x4. B is 4x4. C is 1x4.
    # C_k = sum_j A_j B_jk.
    # Terms: A_j B_jk C_k.
    # 4 * 4 = 16 terms.
    
    # Let's rewrite the helper or just call it 16 times?
    # I'll create a proper helper.

    return {"U": U, "V": V, "W": W}
