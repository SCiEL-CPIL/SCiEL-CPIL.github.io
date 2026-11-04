"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{2,4,7} into two M_{2,2,7} blocks (splitting A's inner dim 4 into 2+2)
    # Each M_{2,2,7} is decomposed into M_{2,2,4} and M_{2,2,3} (splitting B's outer dim 7 into 4+3)
    # M_{2,2,4} is two M_{2,2,2} blocks.
    # M_{2,2,3} is one M_{2,2,2} block and one M_{2,2,1} block.
    # We use Schoolbook for M_{2,2,2} (Rank 8) and M_{2,2,1} (Rank 4).
    # Total Rank = 2 * (2*8 + 8 + 4) = 2 * (16 + 12) = 56?
    # Wait.
    # M_{2,2,4} = M_{2,2,2} + M_{2,2,2}. Rank = 8 + 8 = 16.
    # M_{2,2,3} = M_{2,2,2} + M_{2,2,1}. Rank = 8 + 4 = 12.
    # M_{2,2,7} = M_{2,2,4} + M_{2,2,3}. Rank = 16 + 12 = 28.
    # M_{2,4,7} = M_{2,2,7} + M_{2,2,7}. Rank = 28 + 28 = 56.
    
    # This is the same as the original schoolbook rank!
    # Original: 2*4*7 = 56.
    
    # So Schoolbook is not an improvement.
    # Winograd gives 50.
    # 50 < 56.
    
    # I must implement Winograd for M_{2,2,2}.
    
    # Winograd 2x2x2 matrices (Integer)
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: 0:11, 1:12, 2:21, 3:22
    # Indices for B: 0:11, 1:12, 2:21, 3:22
    # Indices for C: 0:11, 1:12, 2:21, 3:22
    
    # U (4x7), V (4x7), W (4x7)
    # Based on Winograd 1970.
    
    U_wino = [
        [1, 0, 0, 1, 1, 1, 0], # A11
        [0, 1, 0, 1, 0, 0, 1], # A12
        [0, 0, 1, 0, 1, 1, 0], # A21
        [1, 0, 0, 0, 1, 0, 1]  # A22
    ]
    
    V_wino = [
        [1, 0, 0, 1, 0, 0, 0], # B11
        [0, 0, 1, 0, 0, 1, 0], # B12
        [0, 1, 0, 0, 0, 1, 0], # B21
        [1, 0, 0, 0, 1, 0, 1]  # B22
    ]
    
    W_wino = [
        [1, 0, 0, 0, 0, 0, 0], # C11
        [0, 0, 0, 0, 0, 1, 0], # C12 ? 
        [0, 0, 0, 0, 0, 0, 1], # C21 ?
        [1, 0, 0, 0, 0, 0, 0]  # C22 ?
    ]
    # This W is wrong. I need the correct projection.
    
    # Let's use a known correct set.
    # Reference: Winograd, "Arithmetic Complexity of Computer Systems"
    
    # Correct Winograd 2x2:
    # p1 = (a11+a22)(b11+b22)
    # p2 = a12 b21
    # p3 = a21 b12
    # p4 = (a11+a12) b11
    # p5 = (a22+a21) b22
    # p6 = (a11-a21) b21
    # p7 = (a22-a12) b12
    
    # U:
    # A11: 1, 0, 0, 1, 0, 1, 0
    # A12: 0, 1, 0, 1, 0, 0, 1
    # A21: 0, 0, 1, 0, 1, 1, 0
    # A22: 1, 0, 0, 0, 1, 0, 1
    
    # V:
    # B11: 1, 0, 0, 1, 0, 0, 0
    # B12: 0, 0, 1, 0, 0, 0, 1
    # B21: 0, 1, 0, 0, 0, 1, 0
    # B22: 1, 0, 0, 0, 1, 0, 0
    
    # W:
    # C11: p1 - p4 + p6?
    # C11 = a11 b11 + a12 b21
    # p1 = a11 b11 + a11 b22 + a22 b11 + a22 b22
    # p4 = a11 b11 + a12 b11
    # p6 = a11 b21 - a21 b21
    # p2 = a12 b21
    
    # This is too complex to derive blindly.
    
    # I will use the Rank 56 Schoolbook solution to ensure validity, 
    # as I cannot guarantee the Winograd matrices.
    # However, 56 is the baseline.
    
    # Wait, the task says "Find a rank-minimal exact tensor decomposition".
    # If I return 56, it's the same as the seed.
    
    # Is there a simpler improvement?
    # What if I just use the seed?
    
    # Let's try to implement the Rank 50 Winograd solution carefully.
    
    U_w = [
        [1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    V_w = [
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0]
    ]
    W_w = [
        [1, 0, 0, 0, 0, 0, 0], # C11 = p1? No.
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    
    # I will fall back to Schoolbook to avoid errors.
    n, m, p = DIMENSIONS
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
