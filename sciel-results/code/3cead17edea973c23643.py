import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin

    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx, v_idx, w_idx are lists of indices
        # val is the coefficient
        pass

    # Helper to add a rank-1 term
    def add_rank1(u_indices, v_indices, w_indices, coeff=1):
        # u_indices: list of (row, val)
        # v_indices: list of (row, val)
        # w_indices: list of (row, val)
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, val in u_indices:
            u_vec[idx] += val * coeff
        for idx, val in v_indices:
            v_vec[idx] += val * coeff
        for idx, val in w_indices:
            w_vec[idx] += val * coeff
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Strassen 2x2 decomposition
    # A indices: 0,1,2,3 (i*m+j)
    # B indices: 0,1,2,3 (j*p+k)
    # C indices: 0,1,2,3 (k*n+i)
    # Note: In our global indices, these are local to the 2x2 block.
    
    def add_strassen_2x2(a_offset, b_offset, c_offset):
        # a_offset: index of A[0,0]
        # b_offset: index of B[0,0]
        # c_offset: index of C[0,0]
        
        # A indices
        a00 = a_offset
        a01 = a_offset + 1
        a10 = a_offset + 2
        a11 = a_offset + 3
        
        # B indices
        b00 = b_offset
        b01 = b_offset + 1
        b10 = b_offset + 2
        b11 = b_offset + 3
        
        # C indices
        c00 = c_offset
        c01 = c_offset + 1
        c10 = c_offset + 2
        c11 = c_offset + 3
        
        # M1 = (A00+A11)(B00+B11)
        add_rank1([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, -1)]) # M1 contributes to C00 and C11?
        # Wait, C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        
        # We need to add contributions to W.
        # It is easier to add the 7 terms and let W accumulate.
        
        # Term 1: (A00+A11)(B00+B11)
        # W contributions: C00 += 1, C11 += 1
        add_rank1([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])
        
        # Term 2: (A10+A11)B00
        # W contributions: C10 += 1, C11 -= 1
        add_rank1([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, -1)])
        
        # Term 3: A00(B01-B11)
        # W contributions: C01 += 1, C11 += 1
        add_rank1([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, 1)])
        
        # Term 4: A11(B10-B00)
        # W contributions: C00 += 1, C10 += 1
        add_rank1([(a11, 1)], [(b10, 1), (b00, -1)], [(c00, 1), (c10, 1)])
        
        # Term 5: (A00+A01)B11
        # W contributions: C00 -= 1, C01 += 1
        add_rank1([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])
        
        # Term 6: (A10-A00)(B00+B01)
        # W contributions: C11 += 1
        add_rank1([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])
        
        # Term 7: (A01-A11)(B10+B11)
        # W contributions: C00 += 1
        add_rank1([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])

    # 1. A4 B44 (2x4 by 4x4)
    # A4 is cols 0..3 of A. B44 is rows 0..3, cols 0..3 of B.
    # C is cols 0..3 of C.
    
    # Split A4 into AL (cols 0,1) and AR (cols 2,3)
    # Split B44 into B_LL (0,0-1), B_LR (0,2-3), B_RL (2,0-1), B_RR (2,2-3)
    
    # A_L is 2x2 at A[0:2, 0:2]. Offset 0.
    # A_R is 2x2 at A[0:2, 2:4]. Offset 2.
    # B_LL is 2x2 at B[0:2, 0:2]. Offset 0.
    # B_RL is 2x2 at B[2:4, 0:2]. Offset 10.
    # B_LR is 2x2 at B[0:2, 2:4]. Offset 2.
    # B_RR is 2x2 at B[2:4, 2:4]. Offset 12.
    
    # C_L is 2x2 at C[0:2, 0:2]. Offset 0. (k=0,1 -> k*2+i)
    # C_R is 2x2 at C[0:2, 2:4]. Offset 4. (k=2,3 -> k*2+i)
    
    # C_L = A_L B_LL + A_R B_RL
    # A_L B_LL: Strassen at A_off=0, B_off=0, C_off=0
    add_strassen_2x2(0, 0, 0)
    
    # A_R B_RL: Strassen at A_off=2, B_off=10, C_off=0
    # Note: C_off is the same because we are adding to C_L.
    add_strassen_2x2(2, 10, 0)
    
    # C_R = A_L B_LR + A_R B_RR
    # A_L B_LR: Strassen at A_off=0, B_off=2, C_off=4
    add_strassen_2x2(0, 2, 4)
    
    # A_R B_RR: Strassen at A_off=2, B_off=12, C_off=4
    add_strassen_2x2(2, 12, 4)
    
    # 2. A1 B14 (2x1 by 1x4)
    # A1 is col 4 of A. Indices 4, 9.
    # B14 is row 4, cols 0..3 of B. Indices 20, 21, 22, 23.
    # C is cols 0..3 of C.
    # C[i,k] = A[i,4] B[4,k].
    # C indices: k*2 + i.
    
    for k in range(4):
        b_idx = 20 + k
        c_idx_0 = k * 2 + 0
        c_idx_1 = k * 2 + 1
        
        # a * B[4,k] -> C[0,k]
        add_rank1([(4, 1)], [(b_idx, 1)], [(c_idx_0, 1)])
        # b * B[4,k] -> C[1,k]
        add_rank1([(9, 1)], [(b_idx, 1)], [(c_idx_1, 1)])
        
    # 3. A4 B41 (2x4 by 4x1)
    # A4 is cols 0..3 of A.
    # B41 is rows 0..3, col 4 of B. Indices 4, 9, 14, 19.
    # C is col 4 of C. Indices 8, 9.
    # C[i,4] = sum_j A[i,j] B[j,4].
    
    for i in range(2):
        for j in range(4):
            a_idx = i * 5 + j
            b_idx = j * 5 + 4
            c_idx = 4 * 2 + i
            add_rank1([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
            
    # 4. A1 B11 (2x1 by 1x1)
    # A1 is col 4. B11 is row 4, col 4.
    # C[0,4] += A[0,4] B[4,4]
    # C[1,4] += A[1,4] B[4,4]
    
    add_rank1([(4, 1)], [(24, 1)], [(8, 1)])
    add_rank1([(9, 1)], [(24, 1)], [(9, 1)])
    
    # Convert lists to numpy arrays or lists of lists
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for r in range(n*m):
            U[r][c] = U_cols[c][r]
        for r in range(m*p):
            V[r][c] = V_cols[c][r]
        for r in range(p*n):
            W[r][c] = W_cols[c][r]
            
    return {"U": U, "V": V, "W": W}

