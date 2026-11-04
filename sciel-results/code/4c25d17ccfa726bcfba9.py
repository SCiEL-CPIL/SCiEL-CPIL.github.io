"""Matrix multiplication 4x4 by 4x8 via tensor decomposition.
Uses a Kronecker-product structure to get rank 98 (2x Strassen 4x4x4 blocks)."""

import time

DIMENSIONS = [4, 4, 8]


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Strassen 2x2x2 decomposition: rank 7
    # U[i, t], V[j, t], W[k, t] for i,j,k in {0,1}, t in {0..6}
    # C[i,k] = sum_t W[k,t] * (sum_{i',j'} U[i',t] A[i',i] V[j',t] B[j',k])
    # Standard Strassen:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B10-B11)
    # M3 = A11*(B01-B10)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B10)
    # M6 = (A01-A11)*(B01+B11)
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5

    # For 2x2x2: U rows are (A_row, A_col) pairs, V rows are (B_row, B_col) pairs,
    # W rows are (C_row, C_col) pairs.
    # U[i*m2+j, t] where m2=2 (inner dim of A which is 2)
    # Actually for A (2x2) * B (2x2): A has indices (i,j), B has (j,k), C has (i,k)
    # U: 4x7, V: 4x7, W: 4x7
    # U[i*2+j, t], V[j*2+k, t], W[i*2+k, t]

    # Strassen coefficients
    # M0: A*(I+I) * B*(I+I) -> (A00+A11)*(B00+B11)
    # U: A00 coeff 1, A11 coeff 1; V: B00 coeff 1, B11 coeff 1
    # M1: (A10+A11)*B00
    # M2: A00*(B10-B11)
    # M3: A11*(B01-B10)
    # M4: (A00+A01)*B11
    # M5: (A10-A00)*(B00+B10)
    # M6: (A01-A11)*(B01+B11)

    # W (output): C00 = M0+M3-M4+M6, C01 = M2+M4, C10 = M1+M3, C11 = M0-M1+M2+M5
    # C indices: (i,k) -> row i*2+k
    # C00: row 0*2+0 = 0
    # C01: row 0*2+1 = 1
    # C10: row 1*2+0 = 2
    # C11: row 1*2+1 = 3

    # Strassen 2x2x2 U, V, W matrices (each 4x7)
    # U[i*2+j, t]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # M0: (A00+A11)*(B00+B11)
    U2[0][0] = 1  # A00
    U2[3][0] = 1  # A11
    V2[0][0] = 1  # B00
    V2[3][0] = 1  # B11
    W2[0][0] = 1  # C00
    W2[3][0] = 1  # C11

    # M1: (A10+A11)*B00
    U2[2][1] = 1  # A10
    U2[3][1] = 1  # A11
    V2[0][1] = 1  # B00
    W2[2][1] = 1  # C10
    W2[3][1] = -1  # C11

    # M2: A00*(B10-B11)
    U2[0][2] = 1  # A00
    V2[2][2] = 1  # B10
    V2[3][2] = -1  # B11
    W2[1][2] = 1  # C01
    W2[3][2] = 1  # C11

    # M3: A11*(B01-B10)
    U2[3][3] = 1  # A11
    V2[1][3] = 1  # B01
    V2[2][3] = -1  # B10
    W2[0][3] = 1  # C00
    W2[2][3] = 1  # C10

    # M4: (A00+A01)*B11
    U2[0][4] = 1  # A00
    U2[1][4] = 1  # A01
    V2[3][4] = 1  # B11
    W2[1][4] = 1  # C01
    W2[0][4] = -1  # C00

    # M5: (A10-A00)*(B00+B10)
    U2[2][5] = 1  # A10
    U2[0][5] = -1  # A00
    V2[0][5] = 1  # B00
    V2[2][5] = 1  # B10
    W2[3][5] = 1  # C11

    # M6: (A01-A11)*(B01+B11)
    U2[1][6] = 1  # A01
    U2[3][6] = -1  # A11
    V2[1][6] = 1  # B01
    V2[3][6] = 1  # B11
    W2[0][6] = 1  # C00

    # Now build 4x4x4 Strassen by Kronecker product
    # 4x4 A * 4x4 B: split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    # Each block multiplication uses Strassen 2x2x2
    # Total rank for 4x4x4: 7 * 7 = 49

    # Build 4x4x4 decomposition:
    # U4: 16x49, V4: 16x49, W4: 16x49
    # U4[i*4+j, t] where t = t_outer * 7 + t_inner
    # The outer Strassen gives which 2x2 block of A to use (via U2 outer)
    # The inner Strassen gives which entry within the 2x2 block (via U2 inner)

    # Outer Strassen: splits 4x4 into 2x2 blocks
    # Outer A block indices: (bi, bj) where bi,bj in {0,1}
    # U2_outer[bi*2+bj, t_out]
    # Inner Strassen: within each 2x2 block
    # U2_inner[i*2+j, t_in] where i,j are within-block indices

    # Combined: U4[(bi*2+i)*(4) + (bj*2+j), t_out*7 + t_in]
    # Wait, let me be more careful with indexing.
    # A is 4x4, indexed by (row, col) where row in 0..3, col in 0..3
    # Block (bi, bj) contains entries A[bi*2+i, bj*2+j] for i,j in 0..1
    # So global row = bi*2+i, global col = bj*2+j
    # U4[row*4+col, t] = U2_outer[bi*2+bj, t_out] * U2_inner[i*2+j, t_in]

    rank_444 = 49
    U4 = [[0]*rank_444 for _ in range(16)]
    V4 = [[0]*rank_444 for _ in range(16)]
    W4 = [[0]*rank_444 for _ in range(16)]

    for t_out in range(7):
        for t_in in range(7):
            t = t_out * 7 + t_in
            for bi in range(2):
                for bj in range(2):
                    u_outer = U2[bi*2+bj][t_out]
                    if u_outer == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            u_inner = U2[i*2+j][t_in]
                            if u_inner == 0:
                                continue
                            row = bi*2+i
                            col = bj*2+j
                            U4[row*4+col][t] = u_outer * u_inner

    for t_out in range(7):
        for t_in in range(7):
            t = t_out * 7 + t_in
            for bi in range(2):
                for bj in range(2):
                    v_outer = V2[bi*2+bj][t_out]
                    if v_outer == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            v_inner = V2[i*2+j][t_in]
                            if v_inner == 0:
                                continue
                            row = bi*2+i
                            col = bj*2+j
                            V4[row*4+col][t] = v_outer * v_inner

    for t_out in range(7):
        for t_in in range(7):
            t = t_out * 7 + t_in
            for bi in range(2):
                for bj in range(2):
                    w_outer = W2[bi*2+bj][t_out]
                    if w_outer == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            w_inner = W2[i*2+j][t_in]
                            if w_inner == 0:
                                continue
                            row = bi*2+i
                            col = bj*2+j
                            W4[row*4+col][t] = w_outer * w_inner

    # Now build 4x4x8 decomposition:
    # A is 4x4, B is 4x8, C is 8x4
    # Split B into two 4x4 blocks: B_left (cols 0-3), B_right (cols 4-7)
    # C is 8x4: C_top (rows 0-3) comes from A*B_left, C_bottom (rows 4-7) comes from A*B_right
    # Wait, C = A * B where A is 4x4, B is 4x8. C[i,k] = sum_j A[i,j]*B[j,k]
    # C has 8 columns (k=0..7), 4 rows (i=0..3)
    # But the task says W has shape (p*n, rank) = (8*4, rank) = (32, rank)
    # W[k*n+i, c] - so W index is k*n+i where k in 0..7, i in 0..3
    # This means C entries are indexed as (k, i) in the W convention
    # C[k*n+i] = C[k,i]... but C is 4x8 (rows=i, cols=k)
    # So W[k*n+i, c] corresponds to C[i,k] in standard notation
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W row index = k*n + i where k is column index of C (0..7), i is row index (0..3)
    # W[k*n+i, c] * (U^T A V)_c = C[i,k]

    # For the two halves:
    # Left half: B_left = B[:, 0:3], C_left = A * B_left (4x4)
    #   C_left[i, k] for k in 0..3 -> W[k*n+i, c] for left half
    # Right half: B_right = B[:, 4:7], C_right = A * B_right (4x4)
    #   C_right[i, k] for k in 0..3 -> W[(k+4)*n+i, c] for right half

    total_rank = 98  # 49 for left + 49 for right

    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(32)]
    W = [[0]*total_rank for _ in range(32)]

    # Left half (cols 0-3 of B, rows 0-3 of C)
    for c in range(49):
        for r in range(16):
            U[r][c] = U4[r][c]
        for r in range(16):
            V[r][c] = V4[r][c]
        for r in range(16):
            W[r][c] = W4[r][c]

    # Right half (cols 4-7 of B, rows 4-7 of C)
    for c in range(49):
        cc = c + 49
        for r in range(16):
            U[r][cc] = U4[r][c]
            V[r+16][cc] = V4[r][c]
            W[r+16][cc] = W4[r][c]

    return {"U": U, "V": V, "W": W}
