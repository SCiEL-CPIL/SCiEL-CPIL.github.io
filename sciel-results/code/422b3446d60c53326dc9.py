import time

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill


def construct():
    n, m, p = DIMENSIONS
    # Split p=7 into p1=3 and p2=4
    # Use rank-23 decomposition for 3x3x3 part
    # Use schoolbook (rank 36) for 3x3x4 part
    # Total rank: 23 + 36 = 59
    
    rank_333 = 23
    rank_334 = 36
    total_rank = rank_333 + rank_334
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 3x3x3 using rank-23 decomposition
    # The rank-23 decomposition coefficients (from Strassen's work)
    # Each term has coefficients for U[i*m+j], V[j*p+k], W[k*n+i]
    # We need 23 terms with specific rational coefficients
    
    # Standard rank-23 decomposition for 3x3x3
    # Using the formulation where we have 23 bilinear forms
    # Coefficients are in U (9 entries), V (9 entries), W (9 entries)
    
    # The rank-23 decomposition uses these 23 terms:
    # Term 0-7: Basic Strassen-like terms
    # Term 8-22: Additional correction terms
    
    # Let me use a verified rank-23 decomposition
    # From the literature, the 23 terms have the following structure:
    
    r23 = [
        # (U_coeffs, V_coeffs, W_coeffs) where each is dict mapping (i,j) or (j,k) or (k,i) to coefficient
        # Term 0
        ({(0,0): 1, (1,1): 1, (2,2): 1},
         {(0,0): 1, (1,1): 1, (2,2): 1},
         {(0,0): 1, (1,1): 1, (2,2): 1}),
        # Term 1
        ({(0,1): 1, (1,0): 1, (1,2): 1, (2,1): 1},
         {(0,1): 1, (1,0): 1, (1,2): 1, (2,1): 1},
         {(0,1): 1, (1,0): 1, (1,2): 1, (2,1): 1}),
        # Term 2
        ({(0,2): 1, (2,0): 1},
         {(0,2): 1, (2,0): 1},
         {(0,2): 1, (2,0): 1}),
        # Term 3
        ({(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
         {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
         {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        # Term 4
        ({(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1},
         {(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1},
         {(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1}),
        # Term 5
        ({(1,1): 1, (1,2): 1, (2,1): 1, (2,2): 1},
         {(1,1): 1, (1,2): 1, (2,1): 1, (2,2): 1},
         {(1,1): 1, (1,2): 1, (2,1): 1, (2,2): 1}),
        # Term 6
        ({(0,1): 1, (0,2): 1, (1,1): 1, (1,2): 1},
         {(0,1): 1, (0,2): 1, (1,1): 1, (1,2): 1},
         {(0,1): 1, (0,2): 1, (1,1): 1, (1,2): 1}),
        # Term 7
        ({(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1},
         {(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1},
         {(0,0): 1, (0,2): 1, (2,0): 1, (2,2): 1}),
    ]
    
    # The above is not quite right. Let me use a more systematic approach.
    # I'll use the known fact that rank(3,3,3) = 23 and construct it properly.
    # 
    # Actually, let me fall back to a safer approach: use schoolbook for everything
    # but try to find a better decomposition through optimization.
    #
    # For now, let me use schoolbook rank 63 to ensure validity.
    
    # Fallback: schoolbook decomposition
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
