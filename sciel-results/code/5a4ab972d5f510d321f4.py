"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    decomp = _build_strassen_decomp(n, m, p)
    if decomp is not None and _verify(decomp["U"], decomp["V"], decomp["W"], n, m, p):
        return decomp
    return _schoolbook_decomp(n, m, p)

def _schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _build_strassen_decomp(n, m, p):
    """Build rank-37 decomposition: Strassen(2x2x2) + schoolbook(2x1x3) + schoolbook(2x3x4)"""
    if n != 2 or m != 3 or p != 7:
        return None
    
    # Part 1: 2x2x2 Strassen (rank 7) for i∈{0,1}, j∈{0,1}, k∈{0,1}
    # Part 2: 2x1x3 schoolbook (rank 6) for i∈{0,1}, j=2, k∈{0,1,2}
    # Part 3: 2x3x4 schoolbook (rank 24) for i∈{0,1}, j∈{0,1,2}, k∈{3,4,5,6}
    # Total: 7 + 6 + 24 = 37
    
    total_rank = 37
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2x2 (columns 0-6)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # U factors (i, j) where j∈{0,1}
    U[0*3+0] = [1, 0, 1, 0, 1, -1, 0]  # i=0,j=0: M1,M3,M5,M6
    U[0*3+1] = [0, 0, 0, 0, 1, 0, 1]   # i=0,j=1: M5,M7
    U[1*3+0] = [0, 1, 0, 0, 0, 1, 0]   # i=1,j=0: M2,M6
    U[1*3+1] = [1, 1, 1, 1, 0, 0, -1]  # i=1,j=1: M1,M2,M3,M7
    
    # V factors (j, k) where j∈{0,1}, k∈{0,1}
    V[0*7+0] = [1, 1, 0, -1, 0, 1, 0]  # j=0,k=0: M1,M2,M4,M6
    V[0*7+1] = [0, 0, 0, 1, 0, 1, 0]   # j=0,k=1: M4,M6
    V[1*7+0] = [1, 0, 1, 0, 0, 0, 1]   # j=1,k=0: M1,M3,M7
    V[1*7+1] = [1, 0, -1, 0, 1, 0, 1]  # j=1,k=1: M1,M3,M5,M7
    
    # W factors (k, i) where k∈{0,1}, i∈{0,1}
    W[0*2+0] = [1, 0, 0, 1, -1, 0, -1]  # k=0,i=0: M1,M4,M5,M7
    W[1*2+0] = [0, 0, 1, 0, 1, 0, 0]    # k=1,i=0: M3,M5
    W[0*2+1] = [0, 1, 0, 0, 0, 0, 1]    # k=0,i=1: M2,M7
    W[1*2+1] = [1, -1, 1, 0, 0, 1, 0]   # k=1,i=1: M1,M2,M3,M6
    
    # Schoolbook for 2x1x3 (j=2, k=0,1,2) - columns 7-12
    col = 7
    for i in range(2):
        for k in range(3):
            U[i*3+2][col] = 1
            V[2*7+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    # Schoolbook for 2x3x4 (j=0,1,2; k=3,4,5,6) - columns 13-36
    col = 13
    for i in range(2):
        for j in range(3):
            for k in range(3, 7):
                U[i*3+j][col] = 1
                V[j*7+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
