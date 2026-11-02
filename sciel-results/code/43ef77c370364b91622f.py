import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

def _time_ok():
    return (time.time() - START_TIME) < TIME_LIMIT

def construct():
    n, m, p = 3, 5, 5
    
    # Strassen blocks: i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    # 4 disjoint 2x2x2 blocks, each rank 7 instead of 8
    # Remaining terms use schoolbook (rank 1 each)
    
    blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
    ]
    
    num_blocks = len(blocks)
    strassen_rank = num_blocks * 7
    schoolbook_rank = n * m * p - num_blocks * 8  # 75 - 32 = 43
    total_rank = strassen_rank + schoolbook_rank  # 28 + 43 = 71
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
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
    for (i0, j0, k0) in blocks:
        col = add_strassen_block(i0, j0, k0, col)
    
    # Add schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not _time_ok():
                    break
                # Check if this term is covered by a Strassen block
                covered = False
                for (i0, j0, k0) in blocks:
                    if i in (i0, i0+1) and j in (j0, j0+1) and k in (k0, k0+1):
                        covered = True
                        break
                if not covered:
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
