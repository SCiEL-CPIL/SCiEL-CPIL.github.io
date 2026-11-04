import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    
    # Strategy: Use Strassen for 2x2x2 blocks and direct terms for the remainder.
    # The 7x4 product can be decomposed into a 4x4 part and a 3x4 part.
    # 2x4x4 can be done with 4 Strassen blocks? No, 2x4x4 is (2x2)x(4x2)x(4x2)??
    # Let's stick to the proven rank 50 construction from the seed.
    
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Standard Strassen 7-term decomposition for 2x2x2 block
        # Indices: i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
        
        # M1 = (A00+A11)(B00+B11)
        # Contributions to U, V, W
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4 = A11(B00-B10)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # The matrix multiplication is C = A * B where A is 2x4 and B is 4x7.
    # We can view this as two 2x2x4 blocks or similar.
    # The seed program used:
    # Block 1: 2x4x4 (k=0..3) -> 4 Strassen blocks?
    # add_strassen(0, 0, 0) covers i={0,1}, j={0,1}, k={0,1}
    # add_strassen(0, 0, 2) covers i={0,1}, j={0,1}, k={2,3}
    # add_strassen(0, 2, 0) covers i={0,1}, j={2,3}, k={0,1}
    # add_strassen(0, 2, 2) covers i={0,1}, j={2,3}, k={2,3}
    # This covers the 2x4x4 sub-tensor with 4*7 = 28 terms.
    
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Block 2: 2x4x2 (k=4..5) -> 2 Strassen blocks?
    # add_strassen(0, 0, 4) covers i={0,1}, j={0,1}, k={4,5}
    # add_strassen(0, 2, 4) covers i={0,1}, j={2,3}, k={4,5}
    # This covers the 2x4x2 sub-tensor with 2*7 = 14 terms.
    
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # Block 3: 2x4x1 (k=6) -> 8 direct terms
    # C[i,6] = sum_j A[i,j] * B[j,6]
    # Each term A[i,j]B[j,6] is rank 1.
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
            
    assert col == rank, f"Rank mismatch: used {col}, expected {rank}"
    
    return {"U": U, "V": V, "W": W}
