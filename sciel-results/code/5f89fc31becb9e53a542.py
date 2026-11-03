"""Matrix multiplication 3x5 x 5x5: Strassen-like block decomposition for lower rank."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x5 matrix multiplication.
    
    Uses a block decomposition strategy:
    - Split A (3x5) into A1 (3x2) and A2 (3x3)
    - Split B (5x5) into B1 (2x5) and B2 (3x5)
    - C = A1*B1 + A2*B2
    
    For the (3,2,5) blocks, use schoolbook (rank 30 each).
    For the (3,3,5) blocks, use schoolbook (rank 45 each).
    Total: 30 + 45 + 45 + 30 = 150? No, that's wrong.
    
    Actually:
    A1*B1: (3,2,5) -> rank 3*2*5 = 30
    A1*B2: (3,2,5) -> rank 3*2*5 = 30
    A2*B1: (3,3,5) -> rank 3*3*5 = 45
    A2*B2: (3,3,5) -> rank 3*3*5 = 45
    Total: 30+30+45+45 = 150. That's worse than 75!
    
    I need to be more careful. The bilinear rank of (m,n,p) is the number of scalar multiplications.
    For schoolbook, it's m*n*p.
    
    Let me just use the direct schoolbook decomposition with rank 75,
    but structure it clearly.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Schoolbook decomposition
    rank = n * m * p  # 75
    U = [[0.0] * rank for _ in range(n * m)]  # 15 x 75
    V = [[0.0] * rank for _ in range(m * p)]  # 25 x 75
    W = [[0.0] * rank for _ in range(p * n)]  # 15 x 75
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
