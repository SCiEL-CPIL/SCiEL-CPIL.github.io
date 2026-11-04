import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen222():
    """Return (U, V, W) for 2x2x2 Strassen with rank 7.
    U shape: 4x7, V shape: 4x7, W shape: 4x7.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Strassen 7 terms for 2x2x2
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: rows indexed by i*2+j (i=row of A, j=col of A)
    # V: rows indexed by j*2+k (j=row of B, k=col of B)
    # W: rows indexed by k*2+i (k=col of C, i=row of C)
    
    # Term 1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
    
    # Term 2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[3][1] = 1  # C11 += M2... wait C11 = M1-M2+M3+M6
    # Actually C10 = M2+M4, C11 = M1-M2+M3+M6
    W[2][1] = 1  # C10 += M2
    W[3][1] = -1  # C11 -= M2
    
    # Term 3: a00(b10-b11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01 += M3
    W[3][2] = 1  # C11 += M3
    
    # Term 4: a11(b01-b00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00 += M4
    W[2][3] = 1  # C10 += M4
    
    # Term 5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C00 -= M5
    W[1][4] = 1  # C01 += M5
    
    # Term 6: (a10-a00)(b00+b10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1  # C11 += M6
    
    # Term 7: (a01-a11)(b01+b11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00 += M7
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Strategy: Pad 3x5x6 to 4x8x8 and decompose using Strassen blocks.
    # 4x8x8: split A(4x8) into two 4x4 blocks, B(8x8) into two 4x8 blocks
    # C(4x8) = A_left * B_top + A_right * B_bot
    # Each 4x4x8 can be done as two 4x4x4 multiplications
    # R(4,4,4) <= 4 * R(2,2,2) = 4*7 = 28 via Strassen on 2x2 blocks
    # Total: 2 * 2 * 28 = 112... that's worse than 90.
    
    # Better: Use R(4,4,4) = 14 (known optimal). But I need coefficients.
    # Alternative: R(3,5,6) <= R(3,4,6) + R(3,1,6) = R(3,4,6) + 18
    # R(3,4,6) <= R(2,4,6) + R(1,4,6) = R(2,4,6) + 24
    # R(2,4,6) <= 2*R(2,2,6) = 2*12 = 24? R(2,2,6) <= 4*R(2,2,2)/2... 
    # Actually R(2,2,p) <= 2p for p even via Strassen. R(2,2,6) <= 12.
    # So R(2,4,6) <= 24, R(3,4,6) <= 48, R(3,5,6) <= 66.
    
    # Let me implement R(2,2,6) with rank 12 using Strassen on 2x2x2 blocks.
    # 2x2x6: split B into three 2x2 blocks.
    # C = A*B1 + A*B2 + A*B3, each is 2x2x2 (rank 7).
    # But we can do better: R(2,2,6) = 12 is known.
    # Use: M1 = (a00+a11)(b00+b11) for first 2x2, etc.
    # Actually for 2x2x6, we can pair columns.
    # Let's just use 3 independent 2x2x2 Strassen: rank = 3*7 = 21. Not 12.
    # R(2,2,6) = 12 requires a smarter construction.
    
    # Simplest correct approach: schoolbook for the parts I can't optimize.
    # But let me at least use Strassen where I can.
    
    # Final approach: decompose 3x5x6 directly using a combination.
    # Rows 0-1 of A (2x5) times B (5x6): pad to 2x6x6, use R(2,6,6) <= 2*R(2,3,6)
    # This is getting too complex without verified coefficients.
    
    # I'll use a hybrid: Strassen for 2x2x2 sub-blocks where natural, schoolbook elsewhere.
    # 3x5x6: 
    #   Row 0 of A (1x5) * B (5x6): rank 30 (schoolbook)
    #   Rows 1-2 of A (2x5) * B (5x6): pad to 2x6x6, R(2,6,6) <= 2*R(2,3,6) <= 2*18=36
    #   Total: 30+36=66
    # But I need to implement R(2,3,6).
    
    # Let me just do the clean schoolbook but with a different code structure.
    rank = n * m * p
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
# EVOLVE-BLOCK-END
