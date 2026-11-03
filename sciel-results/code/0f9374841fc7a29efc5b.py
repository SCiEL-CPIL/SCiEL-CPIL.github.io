# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 5, 6
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # i_off, j_off, k_off are the starting indices for i, j, k in the 2x2x2 block
        # i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
        
        # M1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        U[i_off*m + j_off][col] += -1
        V[(j_off)*p + (k_off)][col] += 1
        V[(j_off)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M2
        U[i_off*m + j_off][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off)][col] += -1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += -1
        col += 1
        
        # M3
        U[i_off*m + j_off][col] += 1
        U[i_off*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M4
        U[i_off*m + j_off][col] += 1
        V[(j_off)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off)*n + (i_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M5
        U[i_off*m + j_off][col] += 1
        U[(i_off+1)*m + j_off][col] += 1
        V[(j_off)*p + (k_off)][col] += 1
        W[(k_off)*n + (i_off)][col] += -1
        W[(k_off+1)*n + (i_off)][col] += 1
        col += 1
        
        # M6
        U[(i_off+1)*m + j_off][col] += -1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off)*p + (k_off)][col] += 1
        V[(j_off+1)*p + (k_off)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M7
        U[i_off*m + j_off][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off)*n + (i_off)][col] += 1
        col += 1

    def add_rank1(i, j, k_start, k_end):
        # n=1 or 2, m=1, p=k_end-k_start
        # If n=1, i is fixed. If n=2, i varies?
        # The function signature is for a specific block type.
        # Let's make it specific to the blocks we have.
        pass

    # Block 1: A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # i_off=0, j_off=0, k_off=0
    add_strassen(0, 0, 0)
    
    # Block 2: A12 (2x3) * B21 (3x2) -> C11 (2x2)
    # Split j (m) into 2+1.
    # A12a (2x2) * B21a (2x2) -> Strassen
    # i_off=0, j_off=2, k_off=0
    add_strassen(0, 2, 0)
    
    # A12b (2x1) * B21b (1x2) -> Rank 1 for 2x1x2
    # i in {0,1}, j=4, k in {2,3}
    # U[i, j] = 1, V[j, k] = 1, W[k, i] = 1
    for i in range(2):
        for k in range(2, 4):
            U[i*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + i][col] += 1
            col += 1
            
    # Block 3: A11 (2x2) * B12 (2x4) -> C12 (2x4)
    # Split k (p) into 2+2.
    # A11 * B12a (2x2) -> Strassen
    # i_off=0, j_off=0, k_off=2
    add_strassen(0, 0, 2)
    
    # A11 * B12b (2x2) -> Strassen
    # i_off=0, j_off=0, k_off=4
    add_strassen(0, 0, 4)
    
    # Block 4: A12 (2x3) * B22 (3x4) -> C12 (2x4)
    # Split j into 2+1, k into 2+2.
    # A12a (2x2) * B22a (2x2) -> Strassen
    # i_off=0, j_off=2, k_off=2
    add_strassen(0, 2, 2)
    
    # A12a (2x2) * B22b (2x2) -> Strassen
    # i_off=0, j_off=2, k_off=4
    add_strassen(0, 2, 4)
    
    # A12b (2x1) * B22c (1x2) -> Rank 1 for 2x1x2
    # i in {0,1}, j=4, k in {2,3}
    for i in range(2):
        for k in range(2, 4):
            U[i*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + i][col] += 1
            col += 1
            
    # A12b (2x1) * B22d (1x2) -> Rank 1 for 2x1x2
    # i in {0,1}, j=4, k in {4,5}
    for i in range(2):
        for k in range(4, 6):
            U[i*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + i][col] += 1
            col += 1

    # Block 5: A21 (1x2) * B11 (2x2) -> C21 (1x2)
    # Split j into 1+1.
    # A21a (1x1) * B11a (1x2) -> Rank 1 for 1x1x2
    # i=2, j=0, k in {0,1}
    for k in range(2):
        U[2*m + 0][col] += 1
        V[0*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
    # A21b (1x1) * B11b (1x2) -> Rank 1 for 1x1x2
    # i=2, j=1, k in {0,1}
    for k in range(2):
        U[2*m + 1][col] += 1
        V[1*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1

    # Block 6: A22 (1x3) * B21 (3x2) -> C21 (1x2)
    # Split j into 1+1+1.
    # A22a (1x1) * B21a (1x2) -> Rank 1
    # i=2, j=2, k in {0,1}
    for k in range(2):
        U[2*m + 2][col] += 1
        V[2*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
    # A22b (1x1) * B21b (1x2) -> Rank 1
    # i=2, j=3, k in {0,1}
    for k in range(2):
        U[2*m + 3][col] += 1
        V[3*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
    # A22c (1x1) * B21c (1x2) -> Rank 1
    # i=2, j=4, k in {0,1}
    for k in range(2):
        U[2*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1

    # Block 7: A21 (1x2) * B12 (2x4) -> C22 (1x4)
    # Split j into 1+1, k into 2+2.
    # A21a (1x1) * B12a (1x4) -> Rank 1
    # i=2, j=0, k in {2,3,4,5}
    for k in range(2, 6):
        U[2*m + 0][col] += 1
        V[0*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
    # A21b (1x1) * B12b (1x4) -> Rank 1
    # i=2, j=1, k in {2,3,4,5}
    for k in range(2, 6):
        U[2*m + 1][col] += 1
        V[1*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1

    # Block 8: A22 (1x3) * B22 (3x4) -> C22 (1x4)
    # Split j into 1+1+1.
    # A22a (1x1) * B22a (1x4) -> Rank 1
    # i=2, j=2, k in {2,3,4,5}
    for k in range(2, 6):
        U[2*m + 2][col] += 1
        V[2*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
    # A22b (1x1) * B22b (1x4) -> Rank 1
    # i=2, j=3, k in {2,3,4,5}
    for k in range(2, 6):
        U[2*m + 3][col] += 1
        V[3*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
    # A22c (1x1) * B22c (1x4) -> Rank 1
    # i=2, j=4, k in {2,3,4,5}
    for k in range(2, 6):
        U[2*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
