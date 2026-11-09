"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matmul tensor."""
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j2 = b // p
            k = b % p
            if j2 != j:
                continue
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k2 != k or i2 != i:
                    expected = 0
                else:
                    expected = 1
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, -1, -1],
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0],
        [0, 0, -1, -1, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
    ]
    return U, V, W

def _decompose_2x2x3():
    """Decompose 2x2x3 using Strassen on first 2 columns + schoolbook on last.
    Rank = 7 + 4 = 11.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U_school, V_school, W_school = _schoolbook(2, 2, 1)
    
    # Combine: total rank 11
    rank_total = 11
    U = [[0] * rank_total for _ in range(2*2)]
    V = [[0] * rank_total for _ in range(2*3)]
    W = [[0] * rank_total for _ in range(3*2)]
    
    # Strassen part (first 7 columns)
    for a in range(4):
        for r in range(7):
            U[a][r] = U_s[a][r]
    for b in range(6):
        j = b // 3
        k = b % 3
        if k < 2:
            b_part = j * 2 + k
            for r in range(7):
                V[b][r] = V_s[b_part][r]
        # else: 0 for k=2
    for c in range(6):
        k = c // 2
        i = c % 2
        if k < 2:
            c_part = k * 2 + i
            for r in range(7):
                W[c][r] = W_s[c_part][r]
        # else: 0 for k=2
    
    # Schoolbook part (last 4 columns) for the 2x2x1 block at column index 2
    col_offset = 7
    for a in range(4):
        i = a // 2
        j = a % 2
        a_part = i * 2 + j
        for r in range(4):
            U[a][col_offset + r] = U_school[a_part][r]
    for b in range(6):
        j = b // 3
        k = b % 3
        if k == 2:
            b_part = j * 1 + 0
            for r in range(4):
                V[b][col_offset + r] = V_school[b_part][r]
        # else: 0
    for c in range(6):
        k = c // 2
        i = c % 2
        if k == 2:
            c_part = 0 * 2 + i
            for r in range(4):
                W[c][col_offset + r] = W_school[c_part][r]
        # else: 0
    
    return U, V, W

def _decompose_2x3x3():
    """Decompose 2x3x3.
    Split A (2x3) = [A1 (2x2) A2 (2x1)]
    Split B (3x3) = [B11 (2x3); B21 (1x3)]
    C = A1*B11 + A2*B21
    R(2,2,3) = 11, R(2,1,3) = 6. Total = 17.
    """
    U_223, V_223, W_223 = _decompose_2x2x3()
    U_213, V_213, W_213 = _schoolbook(2, 1, 3)
    
    rank_total = 17
    U = [[0] * rank_total for _ in range(2*3)]
    V = [[0] * rank_total for _ in range(3*3)]
    W = [[0] * rank_total for _ in range(3*2)]
    
    # Part 1: A1 (2x2) * B11 (2x3) -> uses _decompose_2x2x3
    for a in range(6):
        i = a // 3
        j = a % 3
        if j < 2:
            a_part = i * 2 + j
            for r in range(11):
                U[a][r] = U_223[a_part][r]
        # else: 0 for j=2
    
    for b in range(9):
        j = b // 3
        k = b % 3
        if j < 2:
            b_part = j * 3 + k
            for r in range(11):
                V[b][r] = V_223[b_part][r]
        # else: 0 for j=2
    
    for c in range(6):
        k = c // 2
        i = c % 2
        c_part = k * 2 + i
        for r in range(11):
            W[c][r] = W_223[c_part][r]
    
    # Part 2: A2 (2x1) * B21 (1x3) -> uses schoolbook(2,1,3)
    col_offset = 11
    for a in range(6):
        i = a // 3
        j = a % 3
        if j == 2:
            a_part = i * 1 + 0
            for r in range(6):
                U[a][col_offset + r] = U_213[a_part][r]
        # else: 0
    
    for b in range(9):
        j = b // 3
        k = b % 3
        if j == 2:
            b_part = 0 * 3 + k
            for r in range(6):
                V[b][col_offset + r] = V_213[b_part][r]
        # else: 0
    
    for c in range(6):
        k = c // 2
        i = c % 2
        c_part = k * 2 + i
        for r in range(6):
            W[c][col_offset + r] = W_213[c_part][r]
    
    return U, V, W

def _decompose_2x2x2_schoolbook():
    """Just return schoolbook for 2x2x2."""
    return _schoolbook(2, 2, 2)

def _decompose_2x3x2():
    """Decompose 2x3x2.
    Split A (2x3) = [A1 (2x2) A2 (2x1)]
    Split B (3x2) = [B11 (2x2); B21 (1x2)]
    C = A1*B11 + A2*B21
    R(2,2,2) = 7 (Strassen), R(2,1,2) = 4. Total = 11.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U_212, V_212, W_212 = _schoolbook(2, 1, 2)
    
    rank_total = 11
    U = [[0] * rank_total for _ in range(2*3)]
    V = [[0] * rank_total for _ in range(3*2)]
    W = [[0] * rank_total for _ in range(2*2)]
    
    # Part 1: A1 (2x2) * B11 (2x2)
    for a in range(6):
        i = a // 3
        j = a % 3
        if j < 2:
            a_part = i * 2 + j
            for r in range(7):
                U[a][r] = U_s[a_part][r]
        # else: 0
    
    for b in range(6):
        j = b // 2
        k = b % 2
        if j < 2:
            b_part = j * 2 + k
            for r in range(7):
                V[b][r] = V_s[b_part][r]
        # else: 0
    
    for c in range(4):
        k = c // 2
        i = c % 2
        c_part = k * 2 + i
        for r in range(7):
            W[c][r] = W_s[c_part][r]
    
    # Part 2: A2 (2x1) * B21 (1x2)
    col_offset = 7
    for a in range(6):
        i = a // 3
        j = a % 3
        if j == 2:
            a_part = i * 1 + 0
            for r in range(4):
                U[a][col_offset + r] = U_212[a_part][r]
        # else: 0
    
    for b in range(6):
        j = b // 2
        k = b % 2
        if j == 2:
            b_part = 0 * 2 + k
            for r in range(4):
                V[b][col_offset + r] = V_212[b_part][r]
        # else: 0
    
    for c in range(4):
        k = c // 2
        i = c % 2
        c_part = k * 2 + i
        for r in range(4):
            W[c][col_offset + r] = W_212[c_part][r]
    
    return U, V, W

def _decompose_4x5x6():
    """Decompose 4x5x6.
    Split: 4=2+2, 5=2+3, 6=3+3
    
    A (4x5) = [A11(2x2) A12(2x3)]
               [A21(2x2) A22(2x3)]
    B (5x6) = [B11(2x3) B12(2x3)]
               [B21(3x3) B22(3x3)]
    C (4x6) = [C11(2x3) C12(2x3)]
               [C21(2x3) C22(2x3)]
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Sub-problems:
    - A11*B11: 2x2 * 2x3 = 2x2x3, rank 11
    - A12*B21: 2x3 * 3x3 = 2x3x3, rank 17
    - A11*B12: 2x2 * 2x3 = 2x2x3, rank 11
    - A12*B22: 2x3 * 3x3 = 2x3x3, rank 17
    - A21*B11: 2x2 * 2x3 = 2x2x3, rank 11
    - A22*B21: 2x3 * 3x3 = 2x3x3, rank 17
    - A21*B12: 2x2 * 2x3 = 2x2x3, rank 11
    - A22*B22: 2x3 * 3x3 = 2x3x3, rank 17
    
    Total rank = 4*11 + 4*17 = 44 + 68 = 112
    """
    n, m, p = 4, 5, 6
    
    # Precompute sub-decompositions
    U_223, V_223, W_223 = _decompose_2x2x3()
    U_233, V_233, W_233 = _decompose_2x3x3()
    
    rank_223 = len(U_223[0])  # 11
    rank_233 = len(U_233[0])  # 17
    
    total_rank = 4 * rank_223 + 4 * rank_233  # 112
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to place a sub-decomposition
    def place(U_sub, V_sub, W_sub, off_i, off_j, off_k, ni, nj, np):
        nonlocal col
        rank_part = len(U_sub[0])
        for i in range(ni):
            for j in range(nj):
                a_full = (off_i + i) * m + (off_j + j)
                a_part = i * nj + j
                for r in range(rank_part):
                    U[a_full][col + r] = U_sub[a_part][r]
        for j in range(nj):
            for k in range(np):
                b_full = (off_j + j) * p + (off_k + k)
                b_part = j * np + k
                for r in range(rank_part):
                    V[b_full][col + r] = V_sub[b_part][r]
        for k in range(np):
            for i in range(ni):
                c_full = (off_k + k) * n + (off_i + i)
                c_part = k * ni + i
                for r in range(rank_part):
                    W[c_full][col + r] = W_sub[c_part][r]
        col += rank_part
    
    # C11 = A11*B11 + A12*B21
    # A11 is at (0,0) size 2x2, B11 is at (0,0) size 2x3
    place(U_223, V_223, W_223, 0, 0, 0, 2, 2, 3)
    # A12 is at (0,2) size 2x3, B21 is at (2,0) size 3x3
    place(U_233, V_233, W_233, 0, 2, 0, 2, 3, 3)
    
    # C12 = A11*B12 + A12*B22
    # A11 is at (0,0) size 2x2, B12 is at (0,3) size 2x3
    place(U_223, V_223, W_223, 0, 0, 3, 2, 2, 3)
    # A12 is at (0,2) size 2x3, B22 is at (2,3) size 3x3
    place(U_233, V_233, W_233, 0, 2, 3, 2, 3, 3)
    
    # C21 = A21*B11 + A22*B21
    # A21 is at (2,0) size 2x2, B11 is at (0,0) size 2x3
    place(U_223, V_223, W_223, 2, 0, 0, 2, 2, 3)
    # A22 is at (2,2) size 2x3, B21 is at (2,0) size 3x3
    place(U_233, V_233, W_233, 2, 2, 0, 2, 3, 3)
    
    # C22 = A21*B12 + A22*B22
    # A21 is at (2,0) size 2x2, B12 is at (0,3) size 2x3
    place(U_223, V_223, W_223, 2, 0, 3, 2, 2, 3)
    # A22 is at (2,2) size 2x3, B22 is at (2,3) size 3x3
    place(U_233, V_233, W_233, 2, 2, 3, 2, 3, 3)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try the block decomposition approach
    if time.time() < deadline - 5:
        U2, V2, W2 = _decompose_4x5x6()
        rank2 = len(U2[0]) if U2 else 0
        if rank2 < best_rank:
            if _verify(U2, V2, W2, n, m, p, rank2):
                best_rank = rank2
                best_result = {"U": U2, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
