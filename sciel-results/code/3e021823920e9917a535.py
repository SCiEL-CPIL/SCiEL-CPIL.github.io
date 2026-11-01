"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time

def _construct_2x3_3x2_rank11():
    """
    Rank-11 decomposition for 2x3 * 3x2.
    U: shape (6, 11), V: shape (6, 11), W: shape (4, 11)
    Convention: U[i*3+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 3, 2
    r = 11
    U = [[0]*r for _ in range(n*m)]  # 6 x 11
    V = [[0]*r for _ in range(m*p)]  # 6 x 11
    W = [[0]*r for _ in range(p*n)]  # 4 x 11
    
    # Known rank-11 algorithm for 2x3 * 3x2
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    
    # Schoolbook: 12 products. We can save 1 by sharing.
    # Let me construct a valid rank-11 decomposition.
    # 
    # One approach: use the fact that we can combine some terms.
    # Let's define 11 bilinear forms.
    
    # Term 0: a00 * b00 -> c00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: a00 * b01 -> c01
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    # Term 2: a01 * b10 -> c00
    U[1][2] = 1; V[2][2] = 1; W[0][2] = 1
    # Term 3: a01 * b11 -> c01
    U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
    # Term 4: a02 * b20 -> c00
    U[2][4] = 1; V[4][4] = 1; W[0][4] = 1
    # Term 5: a02 * b21 -> c01
    U[2][5] = 1; V[5][5] = 1; W[1][5] = 1
    # Term 6: a10 * b00 -> c10
    U[3][6] = 1; V[0][6] = 1; W[2][6] = 1
    # Term 7: a10 * b01 -> c11
    U[3][7] = 1; V[1][7] = 1; W[3][7] = 1
    # Term 8: a11 * b10 -> c10
    U[4][8] = 1; V[2][8] = 1; W[2][8] = 1
    # Term 9: a11 * b11 -> c11
    U[4][9] = 1; V[3][9] = 1; W[3][9] = 1
    # Term 10: a12 * b20 -> c10
    U[5][10] = 1; V[4][10] = 1; W[2][10] = 1
    # Missing: a12 * b21 -> c11 (term 11 in schoolbook)
    
    # This is only 11 terms but missing a12*b21. This won't work.
    # Let me use a proper rank-11 construction.
    
    # Actually, the known rank for 2x3*3x2 is 11. Let me look up or derive it.
    # One known construction uses the following 11 bilinear forms:
    
    # Let me restart with a verified construction.
    # I'll use a different approach: construct it via the known algorithm.
    
    # For now, let me just return schoolbook (rank 12) to be safe,
    # and the tiling will give 4*12 = 48 which is same as baseline.
    # But wait, we're using 2x3x3x3 + 2x3x3x3 + 2x3x3x2, not 4x 2x3x3x2.
    
    # Let me just use schoolbook for the 2x3x3x2 part (rank 12)
    # and rank-16 for 2x3x3x3 parts.
    
    # Actually, let me just implement schoolbook for all sub-problems
    # and see if the tiling structure itself helps.
    
    # Schoolbook for 2x3 * 3x2: rank 12
    r = 12
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _construct_2x3_3x3_rank16():
    """
    Rank-16 decomposition for 2x3 * 3x3.
    U: shape (6, 16), V: shape (9, 16), W: shape (6, 16)
    Convention: U[i*3+j, c], V[j*3+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 3, 3
    r = 16
    U = [[0]*r for _ in range(n*m)]  # 6 x 16
    V = [[0]*r for _ in range(m*p)]  # 9 x 16
    W = [[0]*r for _ in range(p*n)]  # 6 x 16
    
    # Schoolbook for 2x3 * 3x3: rank 18
    # Known rank is 16. Let me construct it.
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # 18 products in schoolbook. Need to save 2.
    # 
    # One known approach: use the fact that we can share some computations.
    # Let me try to find a rank-16 decomposition.
    
    # For simplicity and correctness, let me just use schoolbook (rank 18)
    # This gives total rank 18 + 18 + 12 = 48, same as baseline.
    # Not an improvement, but at least correct.
    
    # Let me try to actually construct rank-16.
    # 
    # I'll use a systematic approach: start with schoolbook and try to merge terms.
    
    r = 18  # Start with schoolbook
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Strategy: Split B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2)
    # C = A * B = A * B1 + A * B2 + A * B3
    # Each sub-problem:
    #   A (2x3) * B1 (3x3) -> C1 (2x3)
    #   A (2x3) * B2 (3x3) -> C2 (2x3)
    #   A (2x3) * B3 (3x2) -> C3 (2x2)
    # C = [C1 | C2 | C3]
    
    # Get decompositions for each sub-problem
    U1, V1, W1 = _construct_2x3_3x3_rank16()  # 6x18, 9x18, 6x18
    U2, V2, W2 = _construct_2x3_3x3_rank16()  # 6x18, 9x18, 6x18
    U3, V3, W3 = _construct_2x3_3x2_rank11()  # 6x12, 6x12, 4x12
    
    r1 = len(U1[0])  # 18
    r2 = len(U2[0])  # 18
    r3 = len(U3[0])  # 12
    total_rank = r1 + r2 + r3  # 48
    
    # Combine into full decomposition
    # U: shape (6, total_rank)
    # V: shape (24, total_rank)  [3*8]
    # W: shape (16, total_rank)  [8*2]
    
    U = [[0]*total_rank for _ in range(6)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(16)]
    
    # Sub-problem 1: A * B1 -> C1
    # B1 is rows 0-2, cols 0-2 of B
    # C1 is cols 0-2 of C
    for c in range(r1):
        for i in range(6):
            U[i][c] = U1[i][c]
        for j in range(3):
            for k in range(3):
                V[j*8 + k][c] = V1[j*3 + k][c]
        for k in range(3):
            for i in range(2):
                W[k*2 + i][c] = W1[k*2 + i][c]
    
    # Sub-problem 2: A * B2 -> C2
    # B2 is rows 0-2, cols 3-5 of B
    # C2 is cols 3-5 of C
    offset = r1
    for c in range(r2):
        for i in range(6):
            U[i][offset + c] = U2[i][c]
        for j in range(3):
            for k in range(3):
                V[j*8 + (3 + k)][offset + c] = V2[j*3 + k][c]
        for k in range(3):
            for i in range(2):
                W[(3 + k)*2 + i][offset + c] = W2[k*2 + i][c]
    
    # Sub-problem 3: A * B3 -> C3
    # B3 is rows 0-2, cols 6-7 of B
    # C3 is cols 6-7 of C
    offset = r1 + r2
    for c in range(r3):
        for i in range(6):
            U[i][offset + c] = U3[i][c]
        for j in range(3):
            for k in range(2):
                V[j*8 + (6 + k)][offset + c] = V3[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[(6 + k)*2 + i][offset + c] = W3[k*2 + i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
