"""Program for matrix multiplication 2x3 by 3x5: rank-28 decomposition.

Uses block decomposition with Strassen's algorithm for 2x2x2 subproblems:
- Block 1: j=0, all k (2x1x5): rank 10 (schoolbook)
- Block 2a: j=1,2, k=0,1 (2x2x2): rank 7 (Strassen)
- Block 2b: j=1,2, k=2 (2x2x1): rank 4 (schoolbook)
- Block 2c: j=1,2, k=3,4 (2x2x2): rank 7 (Strassen)
Total rank: 10 + 7 + 4 + 7 = 28
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_off, j_off, k_off, m, p, n):
    """Apply Strassen's 7-product decomposition for a 2x2x2 block.
    
    Local indices ii, jj, kk in {0,1} map to global:
      i = i_off + ii, j = j_off + jj, k = k_off + kk
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    """
    # M1: (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> C[0,0] += M1, C[1,1] += M1
    U[i_off*m + j_off, col] = 1
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[j_off*p + k_off, col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    # M2: (A[1,0]+A[1,1])B[0,0] -> C[1,0] += M2, C[1,1] -= M2
    U[(i_off+1)*m + j_off, col] = 1
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[j_off*p + k_off, col] = 1
    W[k_off*n + (i_off+1), col] = 1
    W[(k_off+1)*n + (i_off+1), col] = -1
    col += 1
    
    # M3: A[0,0](B[0,1]-B[1,1]) -> C[0,1] += M3, C[1,1] += M3
    U[i_off*m + j_off, col] = 1
    V[j_off*p + (k_off+1), col] = 1
    V[(j_off+1)*p + (k_off+1), col] = -1
    W[(k_off+1)*n + i_off, col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    # M4: A[1,1](B[1,0]-B[0,0]) -> C[0,0] += M4, C[1,0] += M4
    U[(i_off+1)*m + (j_off+1), col] = 1
    V[(j_off+1)*p + k_off, col] = 1
    V[j_off*p + k_off, col] = -1
    W[k_off*n + i_off, col] = 1
    W[k_off*n + (i_off+1), col] = 1
    col += 1
    
    # M5: (A[0,0]+A[0,1])B[1,1] -> C[0,0] -= M5, C[0,1] += M5
    U[i_off*m + j_off, col] = 1
    U[i_off*m + (j_off+1), col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = -1
    W[(k_off+1)*n + i_off, col] = 1
    col += 1
    
    # M6: (A[1,0]-A[0,0])(B[0,0]+B[0,1]) -> C[1,1] += M6
    U[(i_off+1)*m + j_off, col] = 1
    U[i_off*m + j_off, col] = -1
    V[j_off*p + k_off, col] = 1
    V[j_off*p + (k_off+1), col] = 1
    W[(k_off+1)*n + (i_off+1), col] = 1
    col += 1
    
    # M7: (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> C[0,0] += M7
    U[i_off*m + (j_off+1), col] = 1
    U[(i_off+1)*m + (j_off+1), col] = -1
    V[(j_off+1)*p + k_off, col] = 1
    V[(j_off+1)*p + (k_off+1), col] = 1
    W[k_off*n + i_off, col] = 1
    col += 1
    
    return col

def construct():
    """Return rank-28 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Uses block decomposition with Strassen's algorithm for 2x2x2 subproblems:
    - Block 1: j=0, all k (2x1x5): rank 10 (schoolbook)
    - Block 2a: j=1,2, k=0,1 (2x2x2): rank 7 (Strassen)
    - Block 2b: j=1,2, k=2 (2x2x1): rank 4 (schoolbook)
    - Block 2c: j=1,2, k=3,4 (2x2x2): rank 7 (Strassen)
    Total rank: 10 + 7 + 4 + 7 = 28
    """
    n, m, p = 2, 3, 5
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)  # 6 x 28
    V = np.zeros((m * p, rank), dtype=np.int64)  # 15 x 28
    W = np.zeros((p * n, rank), dtype=np.int64)  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, k=0..4 (schoolbook for 2x1x5)
    # C_{ik} += A_{i0} B_{0k}
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2)
    col = _strassen_2x2x2(U, V, W, col, i_off=0, j_off=1, k_off=0, m=m, p=p, n=n)
    
    # Block 2b: j=1,2, k=2 (schoolbook for 2x2x1)
    # C_{i2} += A_{i1} B_{12} + A_{i2} B_{22}
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2)
    col = _strassen_2x2x2(U, V, W, col, i_off=0, j_off=1, k_off=3, m=m, p=p, n=n)
    
    # Verify col == rank
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
