"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # For 2x3x8, we can think of it as 2x3x2 blocks (4 blocks)
    # The bilinear complexity of 2x3x2 is at most 12 (schoolbook)
    # But we can potentially do better.
    
    # Let's use a structured approach:
    # View B as having 8 columns = 4 groups of 2 columns
    # For each group of 2 columns, we compute A * B_group where A is 2x3, B_group is 3x2
    # This gives a 2x2 result.
    
    # For 2x3x2 matrix multiplication (A:2x3, B:3x2 -> C:2x2):
    # Schoolbook: 2*3*2 = 12 multiplications
    # Can we do better? 
    
    # Actually, let's think of this differently.
    # A is 2x3, B is 3x2, C = A*B is 2x2
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # Strassen-like for this: 
    # M1 = (a00+a10)*(b00+b01)
    # M2 = (a01+a11)*(b10+b11)
    # M3 = (a02+a12)*(b20+b21)
    # M4 = (a00-a10)*b00
    # M5 = (a01-a11)*b10
    # M6 = (a02-a12)*b20
    # M7 = a00*b01
    # M8 = a01*b11
    # M9 = a02*b21
    # M10 = a10*b01
    # M11 = a11*b11
    # M12 = a12*b21
    
    # That's 12 again. Let me try another decomposition.
    
    # Actually, for 2x3x2, the minimum rank is likely 12 since 
    # M(2,3,2) >= M(2,2,2) = 7 and M(2,3,2) >= 3*2 = 6 (from each row/col)
    # But it's not obvious we can beat 12.
    
    # Let's just use schoolbook: total rank = 4 * 12 = 48
    # Or we can do the full schoolbook: 2*3*8 = 48
    
    # Let me verify: is there any known result for M(2,3,8)?
    # M(2,3,8) <= M(2,3,2)*4 = 12*4 = 48 (if 2x3x2 has rank 12)
    # M(2,3,8) <= M(2,3,1)*8 = 6*8 = 48
    # M(2,3,8) >= M(2,3,1) * 8 / ... hmm
    
    # Actually, let's just implement the straightforward schoolbook.
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
