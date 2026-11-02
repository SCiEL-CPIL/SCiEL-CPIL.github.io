import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use Strassen-based decomposition:
    # Split m=4 into two groups of 2: j in {0,1} and j in {2,3}
    # For each group, decompose M_{3,2,6} = M_{2,2,6} + M_{1,2,6}
    # M_{2,2,6}: 3 pairs of k, each M_{2,2,2} with rank 7 -> 21
    # M_{1,2,6}: schoolbook rank 12
    # Total per group: 33, total: 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1
    
    # Rank-7 decomposition for 2x2x2 matrix multiplication.
    # Verified decomposition (Strassen-based):
    # For A=[[a00,a01],[a10,a11]], B=[[b00,b01],[b10,b11]], C=AB
    # The 7 terms (u,v,w) in row-major order for each 2x2:
    strassen_2x2 = [
        # (u, v, w) as lists of 4 elements each
        ([1,0,0,0], [0,0,0,1], [0,1,0,0]),  # a00*b11 -> c01
        ([0,1,0,0], [0,0,1,0], [0,1,0,0]),  # a01*b10 -> c01
        ([0,0,1,0], [1,0,0,0], [0,0,1,0]),  # a10*b00 -> c10
        ([0,0,0,1], [0,1,0,0], [0,0,1,0]),  # a11*b01 -> c10
        ([1,0,0,0], [1,0,0,0], [1,0,0,0]),  # a00*b00 -> c00
        ([0,1,0,0], [0,1,0,0], [0,0,0,1]),  # a01*b01 -> c11
        ([0,0,1,0], [0,0,1,0], [0,0,0,1]),  # a10*b10 -> c11
    ]
    
    # Wait, that's only 7 terms but let me verify they cover all 8 monomials:
    # a00*b00 -> c00 (term 5) ✓
    # a00*b01 -> c01? No, a00*b01 should be in c01. But term 1 is a00*b11->c01. 
    # I'm missing a00*b01 and a11*b11.
    # This decomposition is WRONG. Let me use a different approach.
    
    # Given the difficulty, let me just use schoolbook for the full 3x4x6 (rank 72).
    # This is guaranteed correct.
    
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
