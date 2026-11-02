"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    
    Uses the Winograd minimum rank 7 decomposition for 2x2 matrix multiplication,
    applied to the 4x4 matrix multiplication via Strassen-style block decomposition.
    
    For 4x4 * 4x8:
    - Split A (4x4) into four 2x2 blocks
    - Split B (4x8) into four 2x4 blocks (each split into two 2x2 blocks)
    - Use rank-7 Winograd for each 2x2*2x2 multiplication
    - Total rank: 7 * 7 * 2 = 98 (4 Strassen products for 4x4*4x4, times 2 for the 8 columns)
    
    Actually, let's use a simpler approach:
    - View 4x4 * 4x8 as computing 8 independent 4x4 * 4x1 multiplications
    - Each 4x4 * 4x1 can be done with rank 16 (schoolbook)
    - But we can do better using Winograd on 2x2 blocks
    
    Let me use the direct schoolbook approach with rank 128, which is guaranteed correct.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    rank = n * m * p  # 128
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 128
    V = [[0] * rank for _ in range(m * p)]  # 32 x 128
    W = [[0] * rank for _ in range(p * n)]  # 32 x 128
    
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
