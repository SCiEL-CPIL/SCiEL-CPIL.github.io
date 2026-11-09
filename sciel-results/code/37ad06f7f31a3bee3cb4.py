import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    # Helper to add a Strassen term for a 2x2x2 block
    # i0, j0, k0 are the base indices for the 2x2x2 block
    # The block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    def add_strassen(i0, j0, k0):
        nonlocal col
        
        # Map relative indices to global
        # A[i, j] -> U[i*m+j]
        # B[j, k] -> V[j*p+k]
        # C[i, k] -> W[k*n+i]
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11
        U[(i0) * m + (j0)][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        # V: B00, B11
        V[(j0) * p + (k0)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        # W: C00, C11
        W[(k0) * n + (i0)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[(i0+1) * m + (j0)][col] = 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0) * p + (k0)][col] = 1
        W[(k0) * n + (i0+1)][col] = 1
        col += 1
        
        # M3 = A00(B10 - B00)
        U[(i0) * m + (j0)][col] = 1
        V[(j0+1) * p + (k0)][col] = 1
        V[(j0) * p + (k0)][col] = -1
        W[(k0) * n + (i0)][col] = 1
        col += 1
        
        # M4 = A11(B01 + B11)
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0) * p + (k0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[(i0) * m + (j0)][col] = 1
        U[(i0) * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0)][col] = 1
        col += 1
        
        # M6 = (A01 - A00)(B00 + B10)
        U[(i0) * m + (j0+1)][col] = 1
        U[(i0) * m + (j0)][col] = -1
        V[(j0) * p + (k0)][col] = 1
        V[(j0+1) * p + (k0)][col] = 1
        W[(k0) * n + (i0+1)][col] = 1
        col += 1
        
        # M7 = (A10 - A11)(B01 + B11)
        U[(i0+1) * m + (j0)][col] = 1
        U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0) * p + (k0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0)][col] = 1
        col += 1

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0)
    # Schoolbook for j=2 in Block 1 outputs (i in {0,1}, k in {0,1})
    for i in range(2):
        for k in range(2):
            add_schoolbook(i, 2, k)

    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2)
    # Schoolbook for j=2 in Block 2 outputs (i in {0,1}, k in {2,3})
    for i in range(2):
        for k in range(2, 4):
            add_schoolbook(i, 2, k)

    # Block 3: i in {0,1}, j in {0,1,2}, k=4
    for i in range(2):
        for j in range(3):
            add_schoolbook(i, j, 4)

    # Block 4: i=2, j in {0,1,2}, k in {0,1}
    for j in range(3):
        for k in range(2):
            add_schoolbook(2, j, k)

    # Block 5: i=2, j in {0,1,2}, k in {2,3}
    for j in range(3):
        for k in range(2, 4):
            add_schoolbook(2, j, k)

    # Block 6: i=2, j in {0,1,2}, k=4
    for j in range(3):
        add_schoolbook(2, j, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
