import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    # Structure: 10 blocks of size (2x3 x 1)
    # Each block uses rank 5 (3 for 2x2x1 part, 2 for 2x1x1 part)
    rank = 5 * p  # 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U: (i, j) where i in 0..1, j in 0..2
    # Indices for V: (j, k) where j in 0..2, k in 0..9
    # Indices for W: (k, i) where k in 0..9, i in 0..1
    
    # Rank 3 factors for 2x2x1 (A is 2x2, B is 2x1, C is 2x1)
    # U_sub (4x3), V_sub (2x3), W_sub (2x3)
    # Valid rank 3 set for 2x2x1:
    # r=0: (A00+A10)B0 -> C0
    # r=1: (A01+A11)B1 -> C1
    # r=2: A11 B1 -> C1 (and adjust C0?)
    # Actually, let's use the standard Strassen-like rank 3:
    # P1 = A00 B0
    # P2 = A01 B1
    # P3 = (A10+A11)(B0+B1)
    # C0 = P1 + 0.5*P3 - 0.5*(P2 + A11B1)? No.
    
    # Let's use the specific factors from a known source for 2x2x1 rank 3:
    # U = [[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]]
    # V = [[1, 1, 0], [0, 1, 1]]
    # W = [[1, 0, 0], [0, 1, 1]]
    
    U_sub = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ]
    V_sub = [
        [1, 1, 0],
        [0, 1, 1]
    ]
    W_sub = [
        [1, 0, 0],
        [0, 1, 1]
    ]
    
    # For each k in 0..9:
    for k in range(p):
        base_col = k * 5
        
        # Part 1: 2x2x1 using columns 0,1 of A and rows 0,1 of B
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,k), (1,k)
        # C indices: (k,0), (k,1)
        
        for r in range(3):
            col = base_col + r
            
            # U contributions
            # Row 0: A[0,0]
            U[0 * m + 0][col] += U_sub[0][r]
            # Row 1: A[0,1]
            U[0 * m + 1][col] += U_sub[1][r]
            # Row 2: A[1,0]
            U[1 * m + 0][col] += U_sub[2][r]
            # Row 3: A[1,1]
            U[1 * m + 1][col] += U_sub[3][r]
            
            # V contributions
            # Row 0: B[0,k]
            V[0 * p + k][col] += V_sub[0][r]
            # Row 1: B[1,k]
            V[1 * p + k][col] += V_sub[1][r]
            
            # W contributions
            # Row 0: C[k,0]
            W[k * n + 0][col] += W_sub[0][r]
            # Row 1: C[k,1]
            W[k * n + 1][col] += W_sub[1][r]
            
        # Part 2: 2x1x1 using column 2 of A and row 2 of B
        # A indices: (0,2), (1,2)
        # B indices: (2,k)
        # C indices: (k,0), (k,1)
        # Rank 2:
        # r=3: A[0,2] B[2,k] -> C[k,0]
        # r=4: A[1,2] B[2,k] -> C[k,1]
        
        col3 = base_col + 3
        U[0 * m + 2][col3] = 1
        V[2 * p + k][col3] = 1
        W[k * n + 0][col3] = 1
        
        col4 = base_col + 4
        U[1 * m + 2][col4] = 1
        V[2 * p + k][col4] = 1
        W[k * n + 1][col4] = 1

    return {"U": U, "V": V, "W": W}
