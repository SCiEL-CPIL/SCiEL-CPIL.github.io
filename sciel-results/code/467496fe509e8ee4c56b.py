import time

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE

    n, m, p = 3, 4, 8
    rank = 88

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: Schoolbook for i=0
    # C[0, k] = sum_j A[0, j] * B[j, k]
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: Strassen for i=1, 2
    # We treat A[1:3, :] as a 2x4 matrix and B[: , :] as 4x8.
    # We split the inner dimension (columns of A, rows of B) into two halves:
    # Left: cols 0-1 of A, rows 0-1 of B.
    # Right: cols 2-3 of A, rows 2-3 of B.
    #
    # For each half, we have a 2x2x8 multiplication (2 rows of A, 2 rows of B, 8 cols of B).
    # Wait, the dimensions are:
    # A_half: 2x2 (rows 1-2 of A, cols j_start..j_start+1)
    # B_half: 2x8 (rows j_start..j_start+1 of B, cols 0-7)
    # C_half: 2x8 (rows 1-2 of C, cols 0-7)
    #
    # We can decompose 2x2x8 into four 2x2x2 blocks based on the column index k.
    # k = 0,1 -> block 0
    # k = 2,3 -> block 1
    # k = 4,5 -> block 2
    # k = 6,7 -> block 3
    #
    # Each 2x2x2 block uses 7 Strassen terms.
    # Total terms per half = 4 * 7 = 28.
    # Total terms for both halves = 56.
    # Total rank = 32 + 56 = 88.

    # Corrected Strassen matrices for 2x2x2
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    strassen_V = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 0, 1, -1], # M3: B10 - B11
        [-1, 1, 0, 0], # M4: B01 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    strassen_W = [
        [1, 0, 0, 1],  # C00 += M1, C11 += M1
        [0, 0, 1, 0],  # C10 += M2
        [0, 1, 0, 0],  # C01 += M3
        [0, 0, 0, 1],  # C11 += M4
        [0, 0, 1, 0],  # C10 += M5
        [-1, 1, 0, 0], # C00 -= M6, C01 += M6
        [1, 0, 0, 0]   # C00 += M7
    ]
    # Verify W matrix standard Strassen:
    # C00 = M1 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 - M4 + M5  <- Wait, standard is C10 = M2 + M5 - M4?
    # Let's re-verify standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M4 + M5
    # C21 = M2 - M3 + M7
    # C22 = M1 - M2 + M3 + M6
    #
    # My indices:
    # 0:(0,0) -> C00
    # 1:(0,1) -> C01
    # 2:(1,0) -> C10
    # 3:(1,1) -> C11
    #
    # C00 (idx 0) = M1 - M6 + M7
    #   M1: 1, M6: -1, M7: 1
    #   W[0] (for M1) at idx 0: 1
    #   W[5] (for M6) at idx 0: -1
    #   W[6] (for M7) at idx 0: 1
    #   Current W[0][0]=1, W[5][0]=-1, W[6][0]=1. Correct.
    #
    # C01 (idx 1) = M3 + M5
    #   M3: 1, M5: 1
    #   W[2] (for M3) at idx 1: 1
    #   W[4] (for M5) at idx 1: 1
    #   Current W[2][1]=1, W[4][1]=1. Correct.
    #
    # C10 (idx 2) = M2 - M3 + M7
    #   M2: 1, M3: -1, M7: 1
    #   W[1] (for M2) at idx 2: 1
    #   W[2] (for M3) at idx 2: -1
    #   W[6] (for M7) at idx 2: 1
    #   Current W[1][2]=1, W[2][2]=0 (WRONG, should be -1), W[6][2]=0 (WRONG, should be 1).
    #   Let's check my strassen_W definition again.
    #   strassen_W[2] is [0, 1, 0, 0]. So idx 2 is 0.
    #   strassen_W[6] is [1, 0, 0, 0]. So idx 2 is 0.
    #   So C10 gets 0 from M3 and M7.
    #   But C10 should be M2 - M3 + M7.
    #   So W[2][2] should be -1 and W[6][2] should be 1.
    #
    # C11 (idx 3) = M1 - M4 + M5 - M7
    #   M1: 1, M4: -1, M5: 1, M7: -1
    #   W[0] (M1) at idx 3: 1
    #   W[3] (M4) at idx 3: -1
    #   W[4] (M5) at idx 3: 1
    #   W[6] (M7) at idx 3: -1
    #   Current:
    #   W[0][3]=1 (Correct)
    #   W[3][3]=1 (WRONG, should be -1)
    #   W[4][3]=0 (WRONG, should be 1)
    #   W[6][3]=0 (WRONG, should be -1)

    # Let's reconstruct strassen_W correctly.
    strassen_W = [
        [1, 0, 0, 1],  # M1: C00+=1, C11+=1
        [0, 0, 1, 0],  # M2: C10+=1
        [0, 1, -1, 0], # M3: C01+=1, C10-=1
        [0, 0, 0, -1], # M4: C11-=1
        [0, 1, 0, 1],  # M5: C01+=1, C11+=1
        [-1, 0, 0, 0], # M6: C00-=1
        [1, 0, 1, -1]  # M7: C00+=1, C10+=1, C11-=1
    ]

    def add_strassen_block(j_start, k_start):
        nonlocal col
        # j_start: 0 or 2 (global row index in B / col index in A)
        # k_start: 0, 2, 4, 6 (global col index in B / col index in C)
        # Local indices:
        # li: 0, 1 -> global i: 1, 2
        # lj: 0, 1 -> global j: j_start, j_start+1
        # lk: 0, 1 -> global k: k_start, k_start+1
        
        for t in range(7):
            for l in range(4):
                val = strassen_U[t][l]
                if val != 0:
                    li = l // 2
                    lj = l % 2
                    gi = 1 + li
                    gj = j_start + lj
                    U[gi * m + gj][col] += val
            
            for l in range(4):
                val = strassen_V[t][l]
                if val != 0:
                    lj = l // 2
                    lk = l % 2
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk][col] += val
            
            for l in range(4):
                val = strassen_W[t][l]
                if val != 0:
                    li = l // 2
                    lk = l % 2
                    gi = 1 + li
                    gk = k_start + lk
                    W[gk * n + gi][col] += val
            
            col += 1

    # Apply to left half (j=0,1)
    for k_block in range(4):
        k_start = k_block * 2
        add_strassen_block(0, k_start)

    # Apply to right half (j=2,3)
    for k_block in range(4):
        k_start = k_block * 2
        add_strassen_block(2, k_start)

    return {"U": U, "V": V, "W": W}
