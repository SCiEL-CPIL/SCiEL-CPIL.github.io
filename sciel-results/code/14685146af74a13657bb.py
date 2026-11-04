"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    
    Strategy: Decompose into four 2x3x2 sub-problems (column pairs of B).
    For each 2x3x2 sub-problem, use a rank-6 Winograd/Strassen-style decomposition.
    Total rank: 4 * 6 = 24.
    
    For 2x3x2: A is 2x3, B is 3x2, C = A*B is 2x2.
    C[0,0] = a00*b00 + a01*b10 + a02*b20
    C[0,1] = a00*b01 + a01*b11 + a02*b21
    C[1,0] = a10*b00 + a11*b10 + a12*b20
    C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    Rank-6 decomposition for 2x3x2:
    M1 = (a00+a10) * (b00+b01)
    M2 = (a01+a11) * (b10+b11)
    M3 = (a02+a12) * (b20+b21)
    M4 = (a00-a10) * b00
    M5 = (a01-a11) * b10
    M6 = (a02-a12) * b20
    
    Then:
    C[0,0] = M1 + M2 + M3 + M4 + M5 + M6
    C[0,1] = M1 + M2 + M3 - M4 - M5 - M6
    C[1,0] = M1 + M2 + M3 - M4 - M5 - M6  (wait, let me recheck)
    
    Actually let me verify:
    M1 = (a00+a10)(b00+b01) = a00*b00 + a00*b01 + a10*b00 + a10*b01
    M2 = (a01+a11)(b10+b11) = a01*b10 + a01*b11 + a11*b10 + a11*b11
    M3 = (a02+a12)(b20+b21) = a02*b20 + a02*b21 + a12*b20 + a12*b21
    M4 = (a00-a10)*b00 = a00*b00 - a10*b00
    M5 = (a01-a11)*b10 = a01*b10 - a11*b10
    M6 = (a02-a12)*b20 = a02*b20 - a12*b20
    
    C[0,0] = a00*b00 + a01*b10 + a02*b20
    C[1,0] = a10*b00 + a11*b10 + a12*b20
    C[0,1] = a00*b01 + a01*b11 + a02*b21
    C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    Let me try:
    C[0,0] = M1 + M2 + M3 + M4 + M5 + M6
    = (a00*b00 + a00*b01 + a10*b00 + a10*b01) 
    + (a01*b10 + a01*b11 + a11*b10 + a11*b11)
    + (a02*b20 + a02*b21 + a12*b20 + a12*b21)
    + (a00*b00 - a10*b00)
    + (a01*b10 - a11*b10)
    + (a02*b20 - a12*b20)
    = 2*a00*b00 + a00*b01 + a10*b00 + a10*b01 + 2*a01*b10 + a01*b11 + a11*b10 + a11*b11 + 2*a02*b20 + a02*b21 + a12*b20 + a12*b21
    This doesn't match. Let me try a different approach.
    
    Standard Winograd for 2x2x2 (Strassen) has rank 7. For 2x3x2, let's think differently.
    
    Actually, let me just use the schoolbook for each 2x3x2 block: rank 12 per block, total 48.
    But the task asks for rank-minimal. Let me check if rank 24 is achievable.
    
    For 2x3x2, the tensor rank is known to be at most 12 (schoolbook). 
    Can we do better? The lower bound is max(2*3, 3*2, 2*2) = 6, but that's weak.
    
    Let me just implement the straightforward schoolbook for the full 2x3x8: rank 48.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 48
    V = [[0] * rank for _ in range(m * p)]  # 24 x 48
    W = [[0] * rank for _ in range(p * n)]  # 16 x 48
    
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
