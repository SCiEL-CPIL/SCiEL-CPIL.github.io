import time
import random

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _make_strassen_decomp(j_offset, k_offset, col_start, U, V, W, n, m, p):
    # Adds a M_{2,2,2} decomposition for i in {0,1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
    # n=2, m=4, p=5
    # U rows: 2*i + j
    # V rows: 5*j + k
    # W rows: 2*k + i
    
    # Strassen coefficients
    # c=0: M1
    U[0*2 + (j_offset+0)][col_start+0] += 1
    U[1*2 + (j_offset+1)][col_start+0] += 1
    V[(j_offset+0)*5 + (k_offset+0)][col_start+0] += 1
    V[(j_offset+1)*5 + (k_offset+1)][col_start+0] += 1
    W[(k_offset+0)*2 + 0][col_start+0] += 1
    W[(k_offset+1)*2 + 1][col_start+0] += 1
    
    # c=1: M2
    U[0*2 + (j_offset+0)][col_start+1] += 1
    U[0*2 + (j_offset+1)][col_start+1] += 1
    V[(j_offset+1)*5 + (k_offset+1)][col_start+1] += 1
    W[(k_offset+1)*2 + 0][col_start+1] += 1
    
    # c=2: M3
    U[0*2 + (j_offset+0)][col_start+2] += 1
    V[(j_offset+0)*5 + (k_offset+0)][col_start+2] += 1
    V[(j_offset+1)*5 + (k_offset+1)][col_start+2] += -1
    W[(k_offset+1)*2 + 0][col_start+2] += 1
    W[(k_offset+1)*2 + 1][col_start+2] += -1
    
    # c=3: M4
    U[1*2 + (j_offset+1)][col_start+3] += 1
    V[(j_offset+1)*5 + (k_offset+1)][col_start+3] += 1
    V[(j_offset+0)*5 + (k_offset+0)][col_start+3] += -1
    W[(k_offset+0)*2 + 0][col_start+3] += 1
    W[(k_offset+1)*2 + 1][col_start+3] += 1
    
    # c=4: M5
    U[1*2 + (j_offset+0)][col_start+4] += 1
    U[1*2 + (j_offset+1)][col_start+4] += 1
    V[(j_offset+0)*5 + (k_offset+0)][col_start+4] += 1
    W[(k_offset+0)*2 + 0][col_start+4] += -1
    W[(k_offset+0)*2 + 1][col_start+4] += 1
    
    # c=5: M6
    U[0*2 + (j_offset+0)][col_start+5] += 1
    U[1*2 + (j_offset+0)][col_start+5] += -1
    V[(j_offset+0)*5 + (k_offset+1)][col_start+5] += 1
    V[(j_offset+1)*5 + (k_offset+1)][col_start+5] += 1
    W[(k_offset+0)*2 + 1][col_start+5] += 1
    W[(k_offset+1)*2 + 1][col_start+5] += 1
    
    # c=6: M7
    U[0*2 + (j_offset+1)][col_start+6] += 1
    U[1*2 + (j_offset+1)][col_start+6] += -1
    V[(j_offset+0)*5 + (k_offset+0)][col_start+6] += 1
    V[(j_offset+1)*5 + (k_offset+0)][col_start+6] += 1
    W[(k_offset+0)*2 + 0][col_start+6] += 1
    W[(k_offset+1)*2 + 1][col_start+6] += -1
    
    return col_start + 7

def _make_standard_decomp(j_offset, k_offset, col_start, U, V, W, n, m, p):
    # Adds a M_{2,2,1} decomposition for i in {0,1}, j in {j_offset, j_offset+1}, k in {k_offset}
    
    # c=0: A00 B00
    U[0*2 + (j_offset+0)][col_start+0] += 1
    V[(j_offset+0)*5 + (k_offset+0)][col_start+0] += 1
    W[(k_offset+0)*2 + 0][col_start+0] += 1
    
    # c=1: A01 B10
    U[0*2 + (j_offset+1)][col_start+1] += 1
    V[(j_offset+1)*5 + (k_offset+0)][col_start+1] += 1
    W[(k_offset+0)*2 + 0][col_start+1] += 1
    
    # c=2: A10 B00
    U[1*2 + (j_offset+0)][col_start+2] += 1
    V[(j_offset+0)*5 + (k_offset+0)][col_start+2] += 1
    W[(k_offset+0)*2 + 1][col_start+2] += 1
    
    # c=3: A11 B10
    U[1*2 + (j_offset+1)][col_start+3] += 1
    V[(j_offset+1)*5 + (k_offset+0)][col_start+3] += 1
    W[(k_offset+0)*2 + 1][col_start+3] += 1
    
    return col_start + 4

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block L: j in {0, 1}
    # k in {0, 1} -> Strassen
    col = _make_strassen_decomp(0, 0, col, U, V, W, n, m, p)
    # k in {2, 3} -> Strassen
    col = _make_strassen_decomp(0, 2, col, U, V, W, n, m, p)
    # k in {4} -> Standard
    col = _make_standard_decomp(0, 4, col, U, V, W, n, m, p)
    
    # Block R: j in {2, 3}
    # k in {0, 1} -> Strassen
    col = _make_strassen_decomp(2, 0, col, U, V, W, n, m, p)
    # k in {2, 3} -> Strassen
    col = _make_strassen_decomp(2, 2, col, U, V, W, n, m, p)
    # k in {4} -> Standard
    col = _make_standard_decomp(2, 4, col, U, V, W, n, m, p)
    
    if col != rank:
        raise ValueError(f"Column mismatch: {col} != {rank}")
        
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Verification failed")
        
    return {"U": U, "V": V, "W": W}
