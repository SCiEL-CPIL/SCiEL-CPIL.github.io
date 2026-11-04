import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    # Schoolbook decomposition for fallback
    rank_schoolbook = n * m * p

    def schoolbook():
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = rank_schoolbook

    # F(2) algorithm for 2x2x2x2 matrix multiplication
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    # Standard Strassen F(2) coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U (A) coefficients for M1..M7
    F2_U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1],   # a11
    ]
    
    # V (B) coefficients for M1..M7
    F2_V = [
        [1, 1, -1, 0, 0, 1, 0],   # b00
        [0, 0, 0, 1, 0, 0, 1],    # b01
        [0, 0, 1, 0, 0, 1, 0],    # b10
        [1, 0, 0, -1, 1, 0, 1],   # b11
    ]
    
    # W (C) coefficients for M1..M7
    F2_W = [
        [1, 0, 1, -1, 0, 0, -1],  # c00
        [0, 0, 0, 1, 0, 0, 1],    # c01
        [0, 1, -1, 0, 0, 0, 0],   # c10
        [1, -1, 0, 0, 1, 1, 0],   # c11
    ]
    
    # Verify F(2) decomposition
    def verify_F2():
        for a_idx in range(4):
            for b_idx in range(4):
                for c_idx in range(4):
                    total = 0
                    for r in range(7):
                        total += F2_U[a_idx][r] * F2_V[b_idx][r] * F2_W[c_idx][r]
                    a_row, a_col = a_idx // 2, a_idx % 2
                    b_row, b_col = b_idx // 2, b_idx % 2
                    c_row, c_col = c_idx // 2, c_idx % 2
                    expected = 1 if (a_row == c_row and a_col == b_row and b_col == c_col) else 0
                    if abs(total - expected) > 1e-9:
                        return False
        return True

    if not verify_F2():
        return best_result

    # Build 4x4x4x4 using tensor product of F(2) with itself
    # Rank will be 7*7 = 49
    U_44 = [[0] * 49 for _ in range(16)]
    V_44 = [[0] * 49 for _ in range(16)]
    W_44 = [[0] * 49 for _ in range(16)]
    
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    A_idx = i1*8 + i0*4 + j1*2 + j0
                    for r1 in range(7):
                        for r0 in range(7):
                            r = r1*7 + r0
                            U_44[A_idx][r] = F2_U[i1*2 + j1][r1] * F2_U[i0*2 + j0][r0]
    
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    B_idx = j1*8 + j0*4 + k1*2 + k0
                    for r1 in range(7):
                        for r0 in range(7):
                            r = r1*7 + r0
                            V_44[B_idx][r] = F2_V[j1*2 + k1][r1] * F2_V[j0*2 + k0][r0]
    
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    C_idx = k1*8 + k0*4 + i1*2 + i0
                    for r1 in range(7):
                        for r0 in range(7):
                            r = r1*7 + r0
                            W_44[C_idx][r] = F2_W[k1*2 + i1][r1] * F2_W[k0*2 + i0][r0]
    
    # Verify the 4x4x4x4 decomposition
    def verify_4444(U, V, W):
        for a_idx in range(16):
            for b_idx in range(16):
                for c_idx in range(16):
                    total = 0
                    for r in range(49):
                        total += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                    i1, rem = divmod(a_idx, 8)
                    i0, rem = divmod(rem, 4)
                    j1, j0 = divmod(rem, 2)
                    
                    j1b, rem = divmod(b_idx, 8)
                    j0b, rem = divmod(rem, 4)
                    k1, k0 = divmod(rem, 2)
                    
                    k1c, rem = divmod(c_idx, 8)
                    k0c, rem = divmod(rem, 4)
                    i1c, i0c = divmod(rem, 2)
                    
                    expected = 1 if (i1 == i1c and i0 == i0c and j1 == j1b and j0 == j0b and k1 == k1c and k0 == k0c) else 0
                    if abs(total - expected) > 1e-9:
                        return False
        return True

    if not verify_4444(U_44, V_44, W_44):
        return best_result

    # Build full 4x4x4x5 decomposition
    # U: 16 rows (4*4), V: 20 rows (4*5), W: 20 rows (5*4)
    # First 49 terms: 4x4x4x4 part (handles k=0,1,2,3)
    # Next 16 terms: 5th column k=4 (schoolbook)
    
    total_rank = 49 + 16
    U_full = [[0] * total_rank for _ in range(16)]
    V_full = [[0] * total_rank for _ in range(20)]
    W_full = [[0] * total_rank for _ in range(20)]
    
    # Copy 4x4x4x4 part (k=0..3)
    for a_idx in range(16):
        for r in range(49):
            U_full[a_idx][r] = U_44[a_idx][r]
    
    for j in range(4):
        for k in range(4):
            b_idx = j*5 + k
            b_idx_44 = j*4 + k
            for r in range(49):
                V_full[b_idx][r] = V_44[b_idx_44][r]
    
    for k in range(4):
        for i in range(4):
            w_idx = k*4 + i
            w_idx_44 = k*4 + i
            for r in range(49):
                W_full[w_idx][r] = W_44[w_idx_44][r]
    
    # 5th column k=4: C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            r = 49 + i*4 + j
            U_full[i*4+j][r] = 1
            V_full[j*5+4][r] = 1
            W_full[4*4+i][r] = 1
    
    # Verify full decomposition
    def check_decomposition(U, V, W):
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if check_decomposition(U_full, V_full, W_full):
        best_result = {"U": U_full, "V": V_full, "W": W_full}
        best_rank = total_rank
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_result
    
    return best_result
