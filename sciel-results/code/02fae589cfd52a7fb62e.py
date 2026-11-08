"""Program for rank-minimal tensor decomposition of 5x5 matrix multiplication."""
import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    
    # Start with schoolbook decomposition (rank 125)
    rank = n * m * p
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
    
    # Try Strassen-based block decomposition
    # Split 5x5 into blocks: top-left 3x3, top-right 3x2, bottom-left 2x3, bottom-right 2x2
    # A = [[A11(3x3), A12(3x2)], [A21(2x3), A22(2x2)]]
    # B = [[B11(3x3), B12(3x2)], [B21(2x3), B22(2x2)]]
    # C = [[C11(3x3), C12(3x2)], [C21(2x3), C22(2x2)]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For the 3x3 blocks, use schoolbook (27 terms each)
    # For 3x2 blocks, use schoolbook (18 terms each)
    # For 2x3 blocks, use schoolbook (18 terms each)
    # For 2x2 blocks, use Strassen (7 terms instead of 8)
    
    # Let's compute the total rank for this approach:
    # C11 = A11(3x3)*B11(3x3) [27] + A12(3x2)*B21(2x3) [18] = 45 terms
    # C12 = A11(3x3)*B12(3x2) [18] + A12(3x2)*B22(2x2) [7] = 25 terms
    # C21 = A21(2x3)*B11(3x3) [18] + A22(2x2)*B21(2x3) [18] = 36 terms
    # C22 = A21(2x3)*B12(3x2) [18] + A22(2x2)*B22(2x2) [7] = 25 terms
    # Total: 45 + 25 + 36 + 25 = 131 terms
    
    # This is worse than schoolbook (125). Let's try a different split.
    
    # Split into 2x3 and 3x2 blocks:
    # A = [[A11(2x3), A12(2x2)], [A21(3x3), A22(3x2)]]
    # B = [[B11(2x3), B12(2x2)], [B21(3x3), B22(3x2)]]
    # This doesn't work dimensionally.
    
    # Let's try: A = [[A11(2x2), A12(2x3)], [A21(3x2), A22(3x3)]]
    # B = [[B11(2x2), B12(2x3)], [B21(3x2), B22(3x3)]]
    # C = [[C11(2x2), C12(2x3)], [C21(3x2), C22(3x3)]]
    
    # C11 = A11(2x2)*B11(2x2) [7] + A12(2x3)*B21(3x2) [18] = 25 terms
    # C12 = A11(2x2)*B12(2x3) [18] + A12(2x3)*B22(3x3) [27] = 45 terms
    # C21 = A21(3x2)*B11(2x2) [18] + A22(3x3)*B21(3x2) [18] = 36 terms
    # C22 = A21(3x2)*B12(2x3) [18] + A22(3x3)*B22(3x3) [27] = 45 terms
    # Total: 25 + 45 + 36 + 45 = 151 terms
    
    # This is also worse. The schoolbook decomposition is optimal for this block structure.
    
    # Let's try a different approach: use the known lower bounds.
    # The Strassen algorithm gives O(n^2.807) for large n, but for 5x5, 
    # the best known is actually 125 (schoolbook) or potentially less.
    
    # Actually, let's just return the schoolbook decomposition.
    # It's guaranteed to be correct and has rank 125.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
