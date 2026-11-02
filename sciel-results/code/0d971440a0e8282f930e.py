import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_off, j_off, k_off, n, m, p):
    """Apply Strassen decomposition for a 2x2x2 sub-block.
    
    Parameters:
        U, V, W: factor matrices
        col: starting column index
        i_off: offset for i index (0 or 1)
        j_off: offset for j index
        k_off: offset for k index
        n, m, p: dimensions
    
    Returns:
        Next column index
    """
    # M1: (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
    U[i_off*m + j_off][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[j_off*p + k_off][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[k_off*n + i_off][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M2: (A[i1,j0]+A[i1,j1])B[j0,k0]
    U[(i_off+1)*m + j_off][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[j_off*p + k_off][col] = 1
    W[k_off*n + (i_off+1)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = -1
    col += 1
    
    # M3: A[i0,j0](B[j0,k1]-B[j1,k1])
    U[i_off*m + j_off][col] = 1
    V[j_off*p + (k_off+1)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = -1
    W[(k_off+1)*n + i_off][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M4: A[i1,j1](B[j1,k0]-B[j0,k0])
    U[(i_off+1)*m + (j_off+1)][col] = 1
    V[(j_off+1)*p + k_off][col] = 1
    V[j_off*p + k_off][col] = -1
    W[k_off*n + i_off][col] = 1
    W[k_off*n + (i_off+1)][col] = 1
    col += 1
    
    # M5: (A[i0,j0]+A[i0,j1])B[j1,k1]
    U[i_off*m + j_off][col] = 1
    U[i_off*m + (j_off+1)][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[k_off*n + i_off][col] = -1
    W[(k_off+1)*n + i_off][col] = 1
    col += 1
    
    # M6: (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
    U[(i_off+1)*m + j_off][col] = 1
    U[i_off*m + j_off][col] = -1
    V[j_off*p + k_off][col] = 1
    V[j_off*p + (k_off+1)][col] = 1
    W[(k_off+1)*n + (i_off+1)][col] = 1
    col += 1
    
    # M7: (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
    U[i_off*m + (j_off+1)][col] = 1
    U[(i_off+1)*m + (j_off+1)][col] = -1
    V[(j_off+1)*p + k_off][col] = 1
    V[(j_off+1)*p + (k_off+1)][col] = 1
    W[k_off*n + i_off][col] = 1
    col += 1
    
    return col

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication.
    
    Returns:
        dict with 'U', 'V', 'W' factor matrices as lists of lists
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 28 decomposition:
    # - Block 1: j=0, all k (2x1x5), schoolbook rank 10
    # - Block 2: j=1,2, k=0,1 (2x2x2), Strassen rank 7
    # - Block 3: j=1,2, k=2 (2x2x1), schoolbook rank 4
    # - Block 4: j=1,2, k=3,4 (2x2x2), Strassen rank 7
    # Total: 10 + 7 + 4 + 7 = 28
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: j=0, k=0..4 (schoolbook)
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Block 2: j=1,2, k=0,1 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    # Block 3: j=1,2, k=2 (schoolbook, 2x2x1)
    for i in range(n):
        for j in range(1, 3):
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
    
    # Block 4: j=1,2, k=3,4 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Convert to lists of lists for the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
