"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(i, j, k, col, coeff):
        U[i * m + j][col] += coeff
        V[j * p + k][col] += coeff
        W[k * n + i][col] += coeff

    def add_term_neg(i, j, k, col, coeff):
        U[i * m + j][col] -= coeff
        V[j * p + k][col] -= coeff
        W[k * n + i][col] -= coeff

    # Using the 7x7x7 -> 7x7x7 tensor decomposition approach adapted for 3x3x3
    # We use a known rank-23 decomposition for 3x3 matrix multiplication
    # This is based on the fact that 3x3x3 can be embedded in 7x7x7
    
    # Actually, let me use a direct construction. The rank of 3x3 matrix multiplication
    # is known to be between 19 and 23. I'll use a rank-23 decomposition.
    
    # One approach: use the standard Strassen-like decomposition
    # For 3x3, we can think of it as blocks and use recursive decomposition
    
    # Let me use a known explicit rank-23 decomposition
    # Reference: various papers on tensor rank of matrix multiplication
    
    # I'll construct a rank-23 decomposition using the following scheme:
    # Split the 3x3 matrices into 2x2 and 1x1 blocks, use Strassen for the 2x2 part
    # and handle the remaining terms separately.
    
    # A[i,j] * B[j,k] for i,j,k in {0,1,2}
    # Terms:
    # (0,0,0): A[0,0]*B[0,0]
    # (0,0,1): A[0,0]*B[0,1]
    # (0,0,2): A[0,0]*B[0,2]
    # (0,1,0): A[0,1]*B[1,0]
    # (0,1,1): A[0,1]*B[1,1]
    # (0,1,2): A[0,1]*B[1,2]
    # (0,2,0): A[0,2]*B[2,0]
    # (0,2,1): A[0,2]*B[2,1]
    # (0,2,2): A[0,2]*B[2,2]
    # (1,0,0): A[1,0]*B[0,0]
    # (1,0,1): A[1,0]*B[0,1]
    # (1,0,2): A[1,0]*B[0,2]
    # (1,1,0): A[1,1]*B[1,0]
    # (1,1,1): A[1,1]*B[1,1]
    # (1,1,2): A[1,1]*B[1,2]
    # (1,2,0): A[1,2]*B[2,0]
    # (1,2,1): A[1,2]*B[2,1]
    # (1,2,2): A[1,2]*B[2,2]
    # (2,0,0): A[2,0]*B[0,0]
    # (2,0,1): A[2,0]*B[0,1]
    # (2,0,2): A[2,0]*B[0,2]
    # (2,1,0): A[2,1]*B[1,0]
    # (2,1,1): A[2,1]*B[1,1]
    
    # That's 23 terms for the schoolbook minus 4 terms.
    # But we need to combine them properly.
    
    # Let me use a different approach: I'll use the known fact that rank(3x3x3) <= 23
    # and construct an explicit decomposition.
    
    # Using the method from: "The rank of 3x3 matrix multiplication"
    # I'll use a parametrized family and verify it.
    
    # Actually, the simplest approach that is guaranteed to work is rank 27 (schoolbook).
    # But we want rank 23. Let me hardcode a known rank-23 decomposition.
    
    # Here's a rank-23 decomposition for 3x3 matrix multiplication:
    # It uses 23 bilinear forms.
    
    # Term 1: A[0,0]*B[0,0] -> C[0,0]
    add_term(0, 0, 0, 0, 1)
    
    # Term 2: A[0,1]*B[1,0] -> C[0,0] (part of Strassen-like)
    # Let me just use the schoolbook for now and verify rank 27, then optimize.
    
    # Actually, let me just return the schoolbook rank 27 for safety, since
    # I'm not 100% sure about the rank-23 construction from memory.
    
    # Reset and do schoolbook
    for row in U:
        for k in range(rank):
            row[k] = 0
    for row in V:
        for k in range(rank):
            row[k] = 0
    for row in W:
        for k in range(rank):
            row[k] = 0
    
    # Schoolbook is rank 27, but let me just use a valid decomposition
    # I'll use rank 27 for correctness
    return None  # This won't work, let me restart

# Let me restart with a clean approach
def construct():
    n, m, p = DIMENSIONS
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
