"""Rewritten program with distinct source structure for the same rank-71 decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank calculation:
    # 4 Strassen blocks of 2x2x2 (rank 7 each) = 28
    # Remaining terms for i in {0,1}: 18 schoolbook terms
    # All terms for i in {2}: 25 schoolbook terms
    # Total rank = 28 + 18 + 25 = 71
    
    total_rank = 71
    
    # Initialize U, V, W as numpy arrays for efficient addition, then convert
    # U shape: (n*m, rank) -> (15, 71)
    # V shape: (m*p, rank) -> (25, 71)
    # W shape: (p*n, rank) -> (15, 71)
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_idx = 0
    
    # Define the 4 Strassen blocks: (i0, j0, k0)
    # i0 is always 0 (since n=3, we only use i=0,1 for Strassen)
    blocks = [
        (0, 0, 0),  # i:0-1, j:0-1, k:0-1
        (0, 2, 0),  # i:0-1, j:2-3, k:0-1
        (0, 0, 2),  # i:0-1, j:0-1, k:2-3
        (0, 2, 2),  # i:0-1, j:2-3, k:2-3
    ]
    
    for (i0, j0, k0) in blocks:
        # Add 7 Strassen terms for this block
        # Term 1: (A00+A11)(B00+B11)
        U[i0*m + j0, col_idx] += 1
        U[(i0+1)*m + (j0+1), col_idx] += 1
        V[j0*p + k0, col_idx] += 1
        V[(j0+1)*p + (k0+1), col_idx] += 1
        W[k0*n + i0, col_idx] += 1
        W[(k0+1)*n + (i0+1), col_idx] += 1
        col_idx += 1
        
        # Term 2: (A10+A11)B00
        U[(i0+1)*m + j0, col_idx] += 1
        U[(i0+1)*m + (j0+1), col_idx] += 1
        V[j0*p + k0, col_idx] += 1
        W[k0*n + (i0+1), col_idx] += 1
        W[(k0+1)*n + (i0+1), col_idx] += -1
        col_idx += 1
        
        # Term 3: A00(B01-B11)
        U[i0*m + j0, col_idx] += 1
        V[j0*p + (k0+1), col_idx] += 1
        V[(j0+1)*p + (k0+1), col_idx] += -1
        W[(k0+1)*n + i0, col_idx] += 1
        W[(k0+1)*n + (i0+1), col_idx] += 1
        col_idx += 1
        
        # Term 4: A11(B10-B00)
        U[(i0+1)*m + (j0+1), col_idx] += 1
        V[(j0+1)*p + k0, col_idx] += 1
        V[j0*p + k0, col_idx] += -1
        W[k0*n + i0, col_idx] += 1
        W[k0*n + (i0+1), col_idx] += 1
        col_idx += 1
        
        # Term 5: (A00+A01)B11
        U[i0*m + j0, col_idx] += 1
        U[i0*m + (j0+1), col_idx] += 1
        V[(j0+1)*p + (k0+1), col_idx] += 1
        W[k0*n + i0, col_idx] += -1
        W[(k0+1)*n + i0, col_idx] += 1
        col_idx += 1
        
        # Term 6: (A10-A00)(B00+B01)
        U[(i0+1)*m + j0, col_idx] += 1
        U[i0*m + j0, col_idx] += -1
        V[j0*p + k0, col_idx] += 1
        V[j0*p + (k0+1), col_idx] += 1
        W[(k0+1)*n + (i0+1), col_idx] += 1
        col_idx += 1
        
        # Term 7: (A01-A11)(B10+B11)
        U[i0*m + (j0+1), col_idx] += 1
        U[(i0+1)*m + (j0+1), col_idx] += -1
        V[(j0+1)*p + k0, col_idx] += 1
        V[(j0+1)*p + (k0+1), col_idx] += 1
        W[k0*n + i0, col_idx] += 1
        col_idx += 1
    
    # Add schoolbook terms for the remaining indices
    # We need to skip indices covered by the Strassen blocks
    covered = np.zeros((n, m, p), dtype=bool)
    for (i0, j0, k0) in blocks:
        for ii in range(i0, i0+2):
            for jj in range(j0, j0+2):
                for kk in range(k0, k0+2):
                    covered[ii, jj, kk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col_idx] += 1
                    V[j*p + k, col_idx] += 1
                    W[k*n + i, col_idx] += 1
                    col_idx += 1
    
    # Convert numpy arrays to list of lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
