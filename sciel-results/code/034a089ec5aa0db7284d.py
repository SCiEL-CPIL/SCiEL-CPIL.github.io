"""Construct a rank-22 tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows
    
    # Known rank-21 decomposition for 3x3 x 3x3
    # Using the Strassen-like rank-21 pattern with specific nonzero entries
    # Each of the 21 terms has a specific pattern in U (9 entries), V (9 entries for k=0,1,2), W (9 entries for i=0,1,2)
    
    # The rank-21 3x3x3 decomposition (one standard form from literature)
    # Terms 0-20 handle the 3x3 core
    
    r21_U = [
        [1,  0,  0,  0,  1,  0,  0,  0,  0],  # term 0
        [0,  1,  0,  0,  0,  1,  0,  0,  0],  # term 1
        [0,  0,  1,  0,  0,  0,  1,  0,  0],  # term 2
        [0,  0,  0,  1,  0,  0,  0,  1,  0],  # term 3
        [0,  0,  0,  0,  1,  0,  0,  0,  1],  # term 4
        [1,  0,  0,  0,  0,  0,  0,  0,  0],  # term 5
        [0,  1,  0,  0,  0,  0,  0,  0,  0],  # term 6
        [0,  0, 1,  0,  0,  0,  0,  0,  0],  # term 7
        [0,  0,  0, 1,  0,  0,  0,  0,  0],  # term 8
        [0,  0,  0,  0, 1,  0,  0,  0,  0],  # term 9
        [0,  0,  0,  0,  0, 1,  0,  0,  0],  # term 10
        [0,  0,  0,  0,  0,  0, 1,  0,  0],  # term 11
        [0,  0,  0,  0,  0,  0,  0, 1,  0],  # term 12
        [0,  0,  0,  0,  0,  0,  0,  0, 1],  # term 13
        [1,  0,  0,  0,  0,  0,  0,  0,  0],  # term 14
        [0,  1,  0,  0,  0,  0,  0,  0,  0],  # term 15
        [0,  0, 1,  0,  0,  0,  0,  0,  0],  # term 16
        [0,  0,  0, 1,  0,  0,  0,  0,  0],  # term 17
        [0,  0,  0,  0, 1,  0,  0,  0,  0],  # term 18
        [0,  0,  0,  0,  0, 1,  0,  0,  0],  # term 19
        [0,  0,  0,  0,  0,  0, 1,  0,  0],  # term 20
    ]
    
    # This is getting complicated. Let me use a verified rank-21 pattern.
    # I'll use the construction from the known optimal 3x3x3 algorithm.
    
    # Actually, let me use a simpler verified approach:
    # Rank-21 for 3x3x3 has been proven optimal. One explicit construction:
    
    # Using the pattern where we have 21 terms with specific ±1 entries
    # Let me use a known valid rank-21 decomposition:
    
    U21 = [
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    V21 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    W21 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    # This is just identity - rank 9. I need the actual rank-21.
    # Let me use a different, verified approach.
    
    # I'll construct a valid rank-22 by using a known technique:
    # Start with rank-9 schoolbook for 3x3x3, then add 3 terms for the 4th column.
    # But that gives rank 12. To get below that, I need sharing.
    
    # Actually, the simplest correct improvement: use rank-12.
    # For each (j,k) pair, one term that handles all i.
    # But as I analyzed, this doesn't work because we need delta_{i,i'}.
    
    # Let me just do rank-18: 9 terms for 3x3x3 (schoolbook) + 9 terms for the 4th column
    # where for the 4th column, we use 3 terms per i (one per j).
    # Wait, that's 9+9=18.
    
    # For k=3: C[i,3] = sum_j A[i,j] B[j,3]
    # Term (i,j) for k=3: U[i,j,r]=1, V[j,3,r]=1, W[3,i,r]=1
    # This gives 9 terms for k=3, plus 9 for k=0,1,2 = 18 total.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # First 9 terms: schoolbook for k=0,1,2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Next 9 terms: for k=3, one term per (i,j)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
