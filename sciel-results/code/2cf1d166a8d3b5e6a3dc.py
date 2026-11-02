"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-35 tensor decomposition for 2x2 * 2x10 matrix multiplication.
    
    Strategy: Treat the 2x10 matrix as 5 independent 2x2 blocks. For each block,
    apply Strassen's algorithm to reduce the 2x2x2 multiplication from rank 8 to rank 7.
    Total rank = 5 * 7 = 35.
    
    The 2x2 * 2x2 multiplication:
        A = [[a11, a12], [a21, a22]]  (n=2, m=2)
        B = [[b11, b12], [b21, b22]]  (m=2, p=2)
        C = A * B = [[c11, c12], [c21, c22]]
    
    Strassen's 7 products:
        M1 = (a11 + a22)(b11 + b22)
        M2 = (a21 + a22)b11
        M3 = a11(b12 - b22)
        M4 = a22(b21 - b11)
        M5 = (a11 + a12)b22
        M6 = (a21 - a11)(b11 + b12)
        M7 = (a22 - a12)(b21 + b22)
        
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For the 2x10 case, we have:
        A is 2x2: [[a0, a1], [a2, a3]]
        B is 2x10: B[:, 2j:2j+2] for j=0..4 are the 5 blocks
        
    For block j, B_j = [[b_{j,0}, b_{j,1}], [b_{j,2}, b_{j,3}]]
    where b_{j,0} = B[0][2j], b_{j,1} = B[0][2j+1], b_{j,2} = B[1][2j], b_{j,3} = B[1][2j+1]
    
    The output C is 2x10, with C[:, 2j:2j+2] = A * B_j
    Following the trace convention W[k*n+i, r] where k is the output column index (0..9) and i is the output row index (0..1).
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # For each of the 5 blocks, we use 7 Strassen products
    num_blocks = p // 2  # 5 blocks
    rank = num_blocks * 7  # 35
    
    # U has shape (n*m, rank) = (4, 35)
    # V has shape (m*p, rank) = (20, 35)
    # W has shape (p*n, rank) = (20, 35)
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen factors for a single 2x2x2 block
    # A entries: a0=A[0][0], a1=A[0][1], a2=A[1][0], a3=A[1][1]
    # B entries: b0=B[0][0], b1=B[0][1], b2=B[1][0], b3=B[1][1]
    
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2-a0)(b0+b1)
    # M7 = (a3-a1)(b2+b3)
    
    # C11 = M1 + M4 - M5 + M7  -> output index k=0, i=0 -> W[0*2+0] = W[0]
    # C12 = M3 + M5             -> output index k=1, i=0 -> W[1*2+0] = W[2]
    # C21 = M2 + M4             -> output index k=0, i=1 -> W[0*2+1] = W[1]
    # C22 = M1 - M2 + M3 + M6   -> output index k=1, i=1 -> W[1*2+1] = W[3]
    
    # U factors for A (same for all blocks):
    # M1: a0+a3 -> U[0][col] += 1, U[3][col] += 1
    # M2: a2+a3 -> U[2][col] += 1, U[3][col] += 1
    # M3: a0    -> U[0][col] += 1
    # M4: a3    -> U[3][col] += 1
    # M5: a0+a1 -> U[0][col] += 1, U[1][col] += 1
    # M6: a2-a0 -> U[2][col] += 1, U[0][col] += -1
    # M7: a3-a1 -> U[3][col] += 1, U[1][col] += -1
    
    # V factors for B (depend on block j):
    # For block j, B entries are at V indices:
    # b0 = B[0][2j] -> V[0*10 + 2j] = V[2j]
    # b1 = B[0][2j+1] -> V[0*10 + 2j+1] = V[2j+1]
    # b2 = B[1][2j] -> V[1*10 + 2j] = V[10+2j]
    # b3 = B[1][2j+1] -> V[1*10 + 2j+1] = V[10+2j+1]
    
    # M1: b0+b3 -> V[2j][col] += 1, V[10+2j+1][col] += 1
    # M2: b0    -> V[2j][col] += 1
    # M3: b1-b3 -> V[2j+1][col] += 1, V[10+2j+1][col] += -1
    # M4: b2-b0 -> V[10+2j][col] += 1, V[2j][col] += -1
    # M5: b3    -> V[10+2j+1][col] += 1
    # M6: b0+b1 -> V[2j][col] += 1, V[2j+1][col] += 1
    # M7: b2+b3 -> V[10+2j][col] += 1, V[10+2j+1][col] += 1
    
    # W factors for C output (depend on block j):
    # For block j, C entries are at W indices:
    # C11: k=2j, i=0 -> W[2j*2+0] = W[4j]
    # C12: k=2j+1, i=0 -> W[(2j+1)*2+0] = W[4j+2]
    # C21: k=2j, i=1 -> W[2j*2+1] = W[4j+1]
    # C22: k=2j+1, i=1 -> W[(2j+1)*2+1] = W[4j+3]
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    for j in range(num_blocks):
        for s in range(7):
            col = j * 7 + s
            if s == 0:  # M1
                # U
                U[0][col] = 1
                U[3][col] = 1
                # V
                V[2 * j][col] = 1
                V[10 + 2 * j + 1][col] = 1
                # W
                W[4 * j][col] = 1       # C11
                W[4 * j + 3][col] = 1   # C22
            elif s == 1:  # M2
                # U
                U[2][col] = 1
                U[3][col] = 1
                # V
                V[2 * j][col] = 1
                # W
                W[4 * j + 1][col] = 1   # C21
                W[4 * j + 3][col] = -1  # C22
            elif s == 2:  # M3
                # U
                U[0][col] = 1
                # V
                V[2 * j + 1][col] = 1
                V[10 + 2 * j + 1][col] = -1
                # W
                W[4 * j + 2][col] = 1   # C12
                W[4 * j + 3][col] = 1   # C22
            elif s == 3:  # M4
                # U
                U[3][col] = 1
                # V
                V[10 + 2 * j][col] = 1
                V[2 * j][col] = -1
                # W
                W[4 * j][col] = 1       # C11
                W[4 * j + 1][col] = 1   # C21
            elif s == 4:  # M5
                # U
                U[0][col] = 1
                U[1][col] = 1
                # V
                V[10 + 2 * j + 1][col] = 1
                # W
                W[4 * j][col] = -1      # C11
                W[4 * j + 2][col] = 1   # C12
            elif s == 5:  # M6
                # U
                U[2][col] = 1
                U[0][col] = -1
                # V
                V[2 * j][col] = 1
                V[2 * j + 1][col] = 1
                # W
                W[4 * j + 3][col] = 1   # C22
            elif s == 6:  # M7
                # U
                U[3][col] = 1
                U[1][col] = -1
                # V
                V[10 + 2 * j][col] = 1
                V[10 + 2 * j + 1][col] = 1
                # W
                W[4 * j][col] = 1       # C11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

