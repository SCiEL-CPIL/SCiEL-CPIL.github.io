"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # We use a hybrid approach:
    # 1. Apply Strassen's algorithm to disjoint 2x2x2 sub-blocks where possible.
    # 2. Use optimized low-rank decompositions for the remaining "orphan" terms.
    #
    # Dimensions: 3x5 by 5x5
    # i in {0, 1, 2}, j in {0, 1, 2, 3, 4}, k in {0, 1, 2, 3, 4}
    #
    # Strassen Blocks:
    # We can form 2x2x2 blocks using i in {0,1}, j in {j0, j0+1}, k in {k0, k0+1}.
    # Disjoint blocks:
    # 1. i={0,1}, j={0,1}, k={0,1}
    # 2. i={0,1}, j={2,3}, k={0,1}
    # 3. i={0,1}, j={0,1}, k={2,3}
    # 4. i={0,1}, j={2,3}, k={2,3}
    #
    # Total 4 blocks. Each block has rank 7 (Strassen).
    # Strassen Rank = 28.
    #
    # Orphan Terms (not covered by Strassen):
    # 1. i=2, all j, k. (M_1,5,5) -> Rank 5
    # 2. i in {0,1}, j=4, k in {0,1,2,3}. (M_2,1,4) -> Rank 1
    # 3. i in {0,1}, j in {0,1,2,3}, k=4. (M_2,4,1) -> Rank 4
    # 4. i in {0,1}, j=4, k=4. (M_2,1,1) -> Rank 1
    #
    # Total Rank = 28 + 5 + 1 + 4 + 1 = 39.

    total_rank = 39
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block at offsets (i0, j0, k0)
    def add_strassen_block(i0, j0, k0, start_col):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1 = (A00+A11)(B00+B11)
        c = start_col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = start_col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = start_col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = start_col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = start_col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = start_col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = start_col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        
        return start_col + 7

    # Add Strassen blocks
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    for (i0, j0, k0) in blocks:
        col = add_strassen_block(i0, j0, k0, col)
    
    # Add Orphan 1: i=2, all j, k. Rank 5.
    # Terms: A[2,j] * B[j,k]
    # Decomposition: sum_{j=0}^4 A[2,j] (outer) B[j,:]
    for j in range(m):
        c = col
        U[2*m + j][c] += 1
        for k in range(p):
            V[j*p + k][c] += 1
            W[k*n + 2][c] += 1
        col += 1

    # Add Orphan 2: i in {0,1}, j=4, k in {0,1,2,3}. Rank 1.
    # Terms: A[i,4] * B[4,k]
    # Decomposition: A[:,4] (outer) B[4,0:4]
    c = col
    for i in range(2):
        U[i*m + 4][c] += 1
    for k in range(4):
        V[4*p + k][c] += 1
        W[k*n + 0][c] += 1
        W[k*n + 1][c] += 1
    col += 1

    # Add Orphan 3: i in {0,1}, j in {0,1,2,3}, k=4. Rank 4.
    # Terms: A[i,j] * B[j,4]
    # Decomposition: sum_{j=0}^3 A[:,j] (outer) B[j,4]
    for j in range(4):
        c = col
        for i in range(2):
            U[i*m + j][c] += 1
        V[j*p + 4][c] += 1
        W[4*n + 0][c] += 1
        W[4*n + 1][c] += 1
        col += 1

    # Add Orphan 4: i in {0,1}, j=4, k=4. Rank 1.
    # Terms: A[i,4] * B[4,4]
    # Decomposition: A[:,4] (outer) B[4,4]
    c = col
    for i in range(2):
        U[i*m + 4][c] += 1
    V[4*p + 4][c] += 1
    W[4*n + 0][c] += 1
    W[4*n + 1][c] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
