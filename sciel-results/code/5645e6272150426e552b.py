"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Laderman rank-20 decomposition for 3x3*3x3 -> 3x3
    lader_U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
    ]
    lader_V = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
    ]
    lader_W = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
    ]

    rank_lader = 20
    rank_tail = 17
    total_rank = 2 * rank_lader + rank_tail  # 57

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Block 1: k=0..2, cols 0..19
    for r in range(rank_lader):
        for i in range(3):
            for j in range(3):
                U[i*m+j][r] = lader_U[r][i*3+j]
        for j in range(3):
            for k in range(3):
                V[j*p+k][r] = lader_V[r][j*3+k]
        for k in range(3):
            for i in range(3):
                W[k*n+i][r] = lader_W[r][k*3+i]

    # Block 2: k=3..5, cols 20..39
    for r in range(rank_lader):
        for i in range(3):
            for j in range(3):
                U[i*m+j][rank_lader + r] = lader_U[r][i*3+j]
        for j in range(3):
            for k in range(3):
                V[j*p + (3+k)][rank_lader + r] = lader_V[r][j*3+k]
        for k in range(3):
            for i in range(3):
                W[(3+k)*n+i][rank_lader + r] = lader_W[r][k*3+i]

    # Block 3: k=6..7, cols 40..56 (Rank 17)
    col = 2 * rank_lader
    
    # Strassen for A22 * B22
    # A22 indices: i in {0,1}, j in {0,1}
    # B22 indices: j in {0,1}, k in {6,7}
    
    # M1 = (A00+A11)(B06+B17)
    U[0*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*8+6][col] = 1; V[1*8+7][col] = 1
    W[6*3+0][col] = 1; W[7*3+0][col] = 1; W[6*3+1][col] = -1; W[7*3+1][col] = -1 # C00 += M1, C11 -= M1? 
    # Wait, Strassen output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So for M1: W[6,0]+=1, W[7,1]+=1? No, C is 3x2.
    # C[i, k].
    # C00 is i=0, k=6. C01 is i=0, k=7.
    # C10 is i=1, k=6. C11 is i=1, k=7.
    # M1 contributes to C00 and C11.
    W[6*3+0][col] = 1
    W[7*3+1][col] = 1
    col += 1
    
    # M2 = (A10+A11)B06
    U[1*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*8+6][col] = 1
    W[6*3+1][col] = 1 # C10 += M2
    W[7*3+1][col] = -1 # C11 -= M2
    col += 1
    
    # M3 = A00(B07-B17)
    U[0*3+0][col] = 1
    V[0*8+7][col] = 1; V[1*8+7][col] = -1
    W[7*3+0][col] = 1 # C01 += M3
    W[7*3+1][col] = 1 # C11 += M3
    col += 1
    
    # M4 = A11(B16-B06)
    U[1*3+1][col] = 1
    V[1*8+6][col] = 1; V[0*8+6][col] = -1
    W[6*3+0][col] = 1 # C00 += M4
    W[6*3+1][col] = 1 # C10 += M4
    col += 1
    
    # M5 = (A00+A01)B17
    U[0*3+0][col] = 1; U[0*3+1][col] = 1
    V[1*8+7][col] = 1
    W[6*3+0][col] = -1 # C00 -= M5
    W[7*3+0][col] = 1 # C01 += M5
    col += 1
    
    # M6 = (A10-A00)(B06+B07)
    U[1*3+0][col] = 1; U[0*3+0][col] = -1
    V[0*8+6][col] = 1; V[0*8+7][col] = 1
    W[7*3+1][col] = 1 # C11 += M6
    col += 1
    
    # M7 = (A01-A11)(B16+B17)
    U[0*3+1][col] = 1; U[1*3+1][col] = -1
    V[1*8+6][col] = 1; V[1*8+7][col] = 1
    W[6*3+0][col] = 1 # C00 += M7
    col += 1

    # a b' terms
    # P1 = A02 B26 -> C06
    U[0*3+2][col] = 1; V[2*8+6][col] = 1; W[6*3+0][col] = 1
    col += 1
    # P2 = A02 B27 -> C07
    U[0*3+2][col] = 1; V[2*8+7][col] = 1; W[7*3+0][col] = 1
    col += 1
    # P3 = A12 B26 -> C16
    U[1*3+2][col] = 1; V[2*8+6][col] = 1; W[6*3+1][col] = 1
    col += 1
    # P4 = A12 B27 -> C17
    U[1*3+2][col] = 1; V[2*8+7][col] = 1; W[7*3+1][col] = 1
    col += 1

    # b^T B22 terms
    # Q1 = A20 B06 -> C26
    U[2*3+0][col] = 1; V[0*8+6][col] = 1; W[6*3+2][col] = 1
    col += 1
    # Q2 = A21 B16 -> C26
    U[2*3+1][col] = 1; V[1*8+6][col] = 1; W[6*3+2][col] = 1
    col += 1
    # Q3 = A20 B07 -> C27
    U[2*3+0][col] = 1; V[0*8+7][col] = 1; W[7*3+2][col] = 1
    col += 1
    # Q4 = A21 B17 -> C27
    U[2*3+1][col] = 1; V[1*8+7][col] = 1; W[7*3+2][col] = 1
    col += 1

    # c b' terms
    # R1 = A22 B26 -> C26
    U[2*3+2][col] = 1; V[2*8+6][col] = 1; W[6*3+2][col] = 1
    col += 1
    # R2 = A22 B27 -> C27
    U[2*3+2][col] = 1; V[2*8+7][col] = 1; W[7*3+2][col] = 1
    col += 1

    # Verify
    def verify_full(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(total_rank):
                        total += U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i]
                    if total != 1:
                        return False
        return True

    if verify_full(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 72
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
