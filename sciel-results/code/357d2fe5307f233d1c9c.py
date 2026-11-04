import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # Rank 72: Strassen (56) for 4x4x4 part + schoolbook (16) for 4x4x1 part
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 20 x 72
    W = [[0] * rank for _ in range(p * n)]  # 20 x 72
    
    col = 0
    
    # Strassen for 4x4x4 part (cols 0-3 of B, cols 0-3 of C)
    # 8 independent 2x2x2 Strassen blocks (one per term in block multiplication)
    # C_rc = sum_k A_rk B_kc, partitioned into 2x2 blocks
    
    products = [
        (0, 0, 0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 1, 1, 0, 0, 0),  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (1, 0, 0, 0, 1, 0),  # A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
        (1, 1, 1, 0, 1, 0),  # A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
        (0, 0, 0, 1, 0, 1),  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 1, 1, 1, 0, 1),  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
        (1, 0, 0, 1, 1, 1),  # A[2:4,0:2] * B[0:2,2:4] -> C[2:4,2:4]
        (1, 1, 1, 1, 1, 1),  # A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4]
    ]
    
    for (ar, ac, br, bc, cr, cc) in products:
        if time.time() > deadline:
            break
        
        a_base_r = ar * 2
        a_base_c = ac * 2
        b_base_r = br * 2
        b_base_c = bc * 2
        c_base_r = cr * 2
        c_base_c = cc * 2
        
        def a_idx(r, c):
            return (a_base_r + r) * 4 + (a_base_c + c)
        
        def b_idx(r, c):
            return (b_base_r + r) * 5 + (b_base_c + c)
        
        def c_idx(r, c):
            # W row = k*4 + i where k is C column, i is C row
            # c_base_r + r = i (C row), c_base_c + c = k (C column)
            return (c_base_c + c) * 4 + (c_base_r + r)
        
        # M1: (A00+A11)(B00+B11)
        U[a_idx(0,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1   # C00
        W[c_idx(1,1)][col] += 1   # C11
        col += 1
        
        # M2: (A10+A11)B00
        U[a_idx(1,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        W[c_idx(1,0)][col] += 1   # C10
        W[c_idx(1,1)][col] -= 1   # C11 (C11 = M1 - M2 + M3 + M6)
        col += 1
        
        # M3: A00(B01-B11)
        U[a_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] -= 1
        W[c_idx(0,1)][col] += 1   # C01 (C01 = M3 + M5)
        W[c_idx(1,1)][col] += 1   # C11
        col += 1
        
        # M4: A11(B10-B00)
        U[a_idx(1,1)][col] += 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(0,0)][col] -= 1
        W[c_idx(0,0)][col] += 1   # C00 (C00 = M1 + M4 - M5 + M7)
        W[c_idx(1,0)][col] += 1   # C10 (C10 = M2 + M4)
        col += 1
        
        # M5: (A00+A01)B11
        U[a_idx(0,0)][col] += 1
        U[a_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] -= 1   # C00
        W[c_idx(0,1)][col] += 1   # C01
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[a_idx(1,0)][col] += 1
        U[a_idx(0,0)][col] -= 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        W[c_idx(1,1)][col] += 1   # C11
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[a_idx(0,1)][col] += 1
        U[a_idx(1,1)][col] -= 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1   # C00
        col += 1
    
    # Part 2: 4x4x1 schoolbook (rank 16)
    # A (4x4) * B_col4 (4x1) -> C_col4 (4x1)
    # B index: row j, col 4 -> j*5+4
    # C index: row i, col 4 -> W row = 4*4+i = 16+i
    
    for i in range(4):
        for j in range(4):
            if time.time() > deadline:
                break
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1  # W[16+i]
            col += 1
    
    return {"U": U, "V": V, "W": W}
