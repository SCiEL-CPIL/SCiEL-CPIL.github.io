import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Rank = 55
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen parameters for 2x2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    
    # We split m=3 into m1=2 and m2=1.
    # We split p=10 into 5 blocks of size 2.
    
    for b in range(5):
        # Block b corresponds to B columns 2b and 2b+1
        # B row indices are 0 and 1 (since we are in the m1=2 part)
        # B col indices are 2b and 2b+1
        
        # Strassen products
        # P1 = (A00 + A11) * (B00 + B11)
        # P2 = (A10 + A11) * B00
        # P3 = A00 * (B01 - B11)
        # P4 = A11 * (B10 - B00)
        # P5 = (A00 + A01) * B11
        # P6 = (A10 - A00) * (B00 + B01)
        # P7 = (A01 - A11) * (B10 + B11)
        
        # Output contributions
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Map local A indices (i, j) to global (i, j)
        # i is 0 or 1, j is 0 or 1
        # Map local B indices (j, l) to global (j, 2b+l)
        # j is 0 or 1, l is 0 or 1
        
        # We define the 7 rank-1 terms
        
        # Term 1: P1
        # A: A00 + A11
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        # B: B00 + B11 (global cols 2b, 2b+1)
        V[0 * p + (2 * b)][col] = 1
        V[1 * p + (2 * b + 1)][col] = 1
        # W: C00 += 1, C11 += 1
        W[(2 * b) * n + 0][col] = 1   # k=2b, i=0
        W[(2 * b + 1) * n + 1][col] = 1 # k=2b+1, i=1
        col += 1
        
        # Term 2: P2
        # A: A10 + A11
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        # B: B00
        V[0 * p + (2 * b)][col] = 1
        # W: C10 += 1
        W[(2 * b) * n + 1][col] = 1
        col += 1
        
        # Term 3: P3
        # A: A00
        U[0 * m + 0][col] = 1
        # B: B01 - B11
        V[0 * p + (2 * b + 1)][col] = 1
        V[1 * p + (2 * b + 1)][col] = -1
        # W: C01 += 1
        W[(2 * b + 1) * n + 0][col] = 1
        col += 1
        
        # Term 4: P4
        # A: A11
        U[1 * m + 1][col] = 1
        # B: B10 - B00
        V[1 * p + (2 * b)][col] = 1
        V[0 * p + (2 * b)][col] = -1
        # W: C00 += 1, C10 += 1
        W[(2 * b) * n + 0][col] = 1
        W[(2 * b) * n + 1][col] = 1
        col += 1
        
        # Term 5: P5
        # A: A00 + A01
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        # B: B11
        V[1 * p + (2 * b + 1)][col] = 1
        # W: C00 += -1, C01 += 1
        W[(2 * b) * n + 0][col] = -1
        W[(2 * b + 1) * n + 0][col] = 1
        col += 1
        
        # Term 6: P6
        # A: A10 - A00
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        # B: B00 + B01
        V[0 * p + (2 * b)][col] = 1
        V[0 * p + (2 * b + 1)][col] = 1
        # W: C11 += 1
        W[(2 * b + 1) * n + 1][col] = 1
        col += 1
        
        # Term 7: P7
        # A: A01 - A11
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        # B: B10 + B11
        V[1 * p + (2 * b)][col] = 1
        V[1 * p + (2 * b + 1)][col] = 1
        # W: C00 += 1
        W[(2 * b) * n + 0][col] = 1
        col += 1

    # Part 2: j=2
    # A_{i, 2} B_{2, k}
    # i in {0, 1}, k in {0..9}
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
