import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    n, m, p = 3, 3, 8
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 blocks for i in {0,1}, j in {0,1}, k pairs {0,1},{2,3},{4,5},{6,7}
    for k_off in [0, 2, 4, 6]:
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        U[0][col] += 1
        U[4][col] += 1  # i=1, j=1 -> index 1*3+1=4
        V[0][col] += 1
        V[5][col] += 1  # j=1, k=1 -> index 1*8+1=9? No, j*p+k. j=1,k=1 -> 1*8+1=9
        # Wait, let me re-index. V[j*p+k]. For j=0,k=0: 0. j=1,k=1: 1*8+1=9.
        # But we're doing 2x2 sub-block, so within the block:
        # Local indices: i_loc in {0,1}, j_loc in {0,1}, k_loc in {0,1}
        # Global: i=i_off+i_loc, j=j_off+j_loc, k=k_off+k_loc
        # U index: i*m+j = (0+i_loc)*3 + (0+j_loc) = 3*i_loc + j_loc
        # V index: j*p+k = (0+j_loc)*8 + (k_off+k_loc) = 8*j_loc + k_off + k_loc
        # W index: k*n+i = (k_off+k_loc)*3 + (0+i_loc) = 3*(k_off+k_loc) + i_loc
        
        # Let me redo this carefully
        # Actually, let me just use a helper function approach
    
    # Let me restart with a cleaner implementation
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def strassen_222(i0, j0, k0, c0):
        """Add Strassen 2x2x2 terms starting at column c0.
        i0, j0, k0 are the offsets for the 2x2 sub-block.
        U index: (i0+di)*m + (j0+dj)
        V index: (j0+dj)*p + (k0+dk)
        W index: (k0+dk)*n + (i0+di)
        """
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[(i0+0)*m + (j0+0)][c0] += 1
        U[(i0+1)*m + (j0+1)][c0] += 1
        V[(j0+0)*p + (k0+0)][c0] += 1
        V[(j0+1)*p + (k0+1)][c0] += 1
        W[(k0+0)*n + (i0+0)][c0] += 1
        W[(k0+1)*n + (i0+1)][c0] += 1
        # M2 = (A10+A11)B00
        U[(i0+1)*m + (j0+0)][c0+1] += 1
        U[(i0+1)*m + (j0+1)][c0+1] += 1
        V[(j0+0)*p + (k0+0)][c0+1] += 1
        W[(k0+0)*n + (i0+1)][c0+1] += 1
        W[(k0+1)*n + (i0+1)][c0+1] -= 1
        # M3 = A00(B10-B11)
        U[(i0+0)*m + (j0+0)][c0+2] += 1
        V[(j0+1)*p + (k0+0)][c0+2] += 1
        V[(j0+1)*p + (k0+1)][c0+2] -= 1
        W[(k0+0)*n + (i0+0)][c0+2] += 1
        W[(k0+1)*n + (i0+0)][c0+2] += 1
        # M4 = A11(B01-B00)
        U[(i0+1)*m + (j0+1)][c0+3] += 1
        V[(j0+0)*p + (k0+1)][c0+3] += 1
        V[(j0+0)*p + (k0+0)][c0+3] -= 1
        W[(k0+0)*n + (i0+0)][c0+3] += 1
        W[(k0+0)*n + (i0+1)][c0+3] += 1
        # M5 = (A00+A01)B11
        U[(i0+0)*m + (j0+0)][c0+4] += 1
        U[(i0+0)*m + (j0+1)][c0+4] += 1
        V[(j0+1)*p + (k0+1)][c0+4] += 1
        W[(k0+0)*n + (i0+0)][c0+4] -= 1
        W[(k0+1)*n + (i0+0)][c0+4] += 1
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + (j0+0)][c0+5] += 1
        U[(i0+0)*m + (j0+0)][c0+5] -= 1
        V[(j0+0)*p + (k0+0)][c0+5] += 1
        V[(j0+0)*p + (k0+1)][c0+5] += 1
        W[(k0+1)*n + (i0+1)][c0+5] += 1
        # M7 = (A01-A11)(B10+B11)
        U[(i0+0)*m + (j0+1)][c0+6] += 1
        U[(i0+1)*m + (j0+1)][c0+6] -= 1
        V[(j0+1)*p + (k0+0)][c0+6] += 1
        V[(j0+1)*p + (k0+1)][c0+6] += 1
        W[(k0+0)*n + (i0+0)][c0+6] += 1
        W[(k0+1)*n + (i0+0)][c0+6] -= 1
    
    # Add 4 Strassen blocks
    for k_off in [0, 2, 4, 6]:
        strassen_222(0, 0, k_off, col)
        col += 7
    
    # Schoolbook for i=2, j in {0,1,2}, k in {0..7}: 3*8 = 24 terms
    for j in range(3):
        for k in range(8):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    # Schoolbook for i in {0,1}, j=2, k in {0..7}: 2*8 = 16 terms
    for i in range(2):
        for k in range(8):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if total != 1:
                        return False
        return True
    
    if verify():
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][c] = 1
                V_sb[j*p + k][c] = 1
                W_sb[k*n + i][c] = 1
                c += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
